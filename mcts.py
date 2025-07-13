from __future__ import annotations

import copy
from enum import Enum, auto
import math
import time
from typing import Callable, Iterable, Mapping, Optional, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sys

from alphago_nn import AlphaZeroNet
from board import Board, GameResult
from five_in_a_row_board import FiveInARowBoard
from model_manager import ModelCheckpointManager
from training_sample_queue import ReplayBuffer, SelfPlayGameBuffer

def select_action(N: np.ndarray, policy: np.ndarray) -> Optional[int]:
    max_value = np.max(N)
    max_indices = np.flatnonzero(N == max_value)

    if len(max_indices) == 0:
        return None  # No legal move
    elif len(max_indices) == 1:
        return max_indices[0]
    else:
        
        best_index = max_indices[0]
        best_policy = policy[best_index]

        for idx in max_indices[1:]:
            if policy[idx] > best_policy:
                best_index = idx
                best_policy = policy[idx]
        print(f"Tie breaking: {best_index}")
        return best_index

class MCTSNode:
    def __init__(self, children_size: int, policy_func: Callable[[np.ndarray], Tuple[Iterable[np.ndarray], Iterable[float]]], 
                 board: Board, root_to_play: int, parent: MCTSNode = None, last_action: Optional[int] = None):
        self.children_W = np.zeros(children_size, dtype=np.float32)
        self.children_N = np.zeros(children_size, dtype=np.float32)
        self.children_Q = np.zeros(children_size, dtype=np.float32)
        self.N = 1
        self.children_index: Mapping[int, MCTSNode] = {}
        self.parent = parent
        self.root_to_play = root_to_play
        self.to_play = board.get_current_player()
        self.policy_func = policy_func
        self.board = board
        self.children_size = children_size
        self.game_result = board.get_game_result(root_to_play);
        self.legal_actions = self.board.get_legal_moves().flatten();
        self.legal_actions = np.append(self.legal_actions, False);
        self.illegal_actions = np.logical_not(self.legal_actions)

        self.last_action: Optional[int] = last_action 
        self.formula_dirty = True
        input = self.board.get_stack()
        winner = board.get_winner()
        if winner is not None:
            if winner == board.get_current_player():
                self.v = 1.0
            else:
                self.v = -1.0
        else:
            policy_arr, v_arr = self.policy_func(input)
            v = v_arr[0];
            policy = policy_arr[0];
            self.v = v
            self.policy = policy
    
    def add_noise(self, alpha: float = 0.03):
        num_legal_actions = np.sum(self.legal_actions)
        alphas = np.ones_like(self.policy) * alpha
        noise = num_legal_actions * np.random.dirichlet(alphas)
        print(f"Noise: {noise}")
        self.policy = 0.75 * self.policy + 0.25 * noise
    
    def get_v(self):
        return self.v

    def pick_next_move(self) -> int:
        return select_action(self.children_N, self.policy)
        # num_mi = self.children_N[np.nonzero(self.children_N)]
        # mi = np.min(self.children_N[np.nonzero(self.children_N)])
        # ma = np.max(self.children_N)
        # if mi == ma and len(num_mi) > 1:
        #     action = np.argmax(self.policy)
        #     print(f"Tie breaking using policy!!!! {action}")
        #     node, count = self.expand_until_leaf_or_terminal(1600, 1, action)
        #     node.back_update()
        #     return action
        # return np.argmax(self.children_N)
    
    def commit_next_move(self) -> MCTSNode:
        action = self.pick_next_move()
        child_node = self.children_index[action]
        child_node.reset_as_root();
        return child_node

    def partial_reset(self):
        self.children_W = np.zeros(self.children_size, dtype=np.float32)
        self.children_N = np.zeros(self.children_size, dtype=np.float32)
        self.children_Q = np.zeros(self.children_size, dtype=np.float32)
        self.children_index: Mapping[int, MCTSNode] = {}
    
    def reset_as_root(self):
        self.partial_reset()
        self.parent = None
        self.root_to_play = self.board.get_current_player()
        self.game_result = self.board.get_game_result(self.root_to_play);
    
    def set_formula_dirty(self): 
        self.formula_dirty = True 
    
    def calc_formula(self, c: float):
        if self.formula_dirty:
            self.formula = self.children_Q + c * self.policy * np.sqrt(self.N) / (1 + self.children_N)
            self.formula[self.illegal_actions] = -1000
            self.action = np.argmax(self.formula)
            self.formula_dirty = False
        else:
            print("not dirty!")
        return (self.formula, self.action)

    def get_training_pi(self, temperature: float) -> np.ndarray:
        N = np.array(self.children_N, dtype=np.float32)
        N_power = np.power(N, 1.0 / temperature)
        total = np.sum(N_power)

        if total == 0:
            return np.full_like(N, fill_value=1.0 / N.size)

        pi = N_power / total
        return pi

    def expand(self, c: float, action_fix = None) -> Tuple[MCTSNode, bool]:
        policy = self.policy

        # formula = self.children_Q + c * policy * np.sqrt(self.N) / (1 + self.children_N)
        # legal_actions = self.board.get_legal_moves().flatten();
        # legal_actions = np.append(legal_actions, False);
        # illegal_actions = self.illegal_actions
        # formula[illegal_actions] = -1000
        # print(f"{formula}\n")
        # formula = self.calc_formula(1)
        if action_fix is not None:
            action = action_fix
        else:
            formula, action = self.calc_formula(1)
        row, col = self.board.unflatten_index(action)


        is_new_node = False
        if action in self.children_index:
            new_node = self.children_index[action]
        else:
            new_board = copy.deepcopy(self.board)
            new_board.play_step(row, col)
            new_node = MCTSNode(self.children_size, self.policy_func, new_board, self.root_to_play, self, action)
            self.children_index[action] = new_node
            is_new_node = True

        return (new_node, is_new_node)
    
    def simulate_one_step_and_make_move(self, sim_count: int) -> MCTSNode:
        start_time = time.time()
        self.add_noise()
        while sim_count > 0:
            node, count = self.expand_until_leaf_or_terminal(sim_count, 1)
            sim_count -= count
            node.back_update()
        end_time = time.time()
        print(self.children_N[:-1].reshape(size, size))
        print(self.children_Q[:-1].reshape(size, size))
        print(self.formula[:-1].reshape(size, size))
        print(self.policy[:-1].reshape(size, size))
        print(f"To play {self.get_board().get_current_player()}, V: {self.get_v()}")
        next_node = self.commit_next_move()
        b = next_node.get_board().render();
        print(f"===<commited one move inference> time {end_time - start_time}=====")
        print(b)
        return next_node

    def expand_until_leaf_or_terminal(self, limit: int, c: float, action_fix = None) -> Tuple[MCTSNode, int]:
        tmp = self
        steps = 1
        while limit > 0:
            if tmp.get_result() is not GameResult.UNDECIDED:
                return (tmp, steps)
            next, is_new_node = tmp.expand(c, action_fix)
            if is_new_node:
                return (next, steps)
            tmp = next
            limit -= 1
            steps += 1
        return (tmp, steps)
    
    def back_update(self):
        tmp_child = self
        v = self.v
        tmp_parent = self.parent
        # print(f"start back! {v}")
        while tmp_parent is not None:
            tmp_parent.children_N[tmp_child.last_action] += 1
            tmp_parent.N += 1
            if tmp_parent.to_play == self.to_play:
                # print("+")
                tmp_parent.children_W[tmp_child.last_action] += v
            else:
                # print("-")
                tmp_parent.children_W[tmp_child.last_action] -= v
            tmp_parent.children_Q[tmp_child.last_action] = tmp_parent.children_W[tmp_child.last_action] / tmp_parent.children_N[tmp_child.last_action]
            tmp_parent.set_formula_dirty()
            tmp_child = tmp_parent
            tmp_parent = tmp_parent.parent
        # print("end back!")

    def get_result(self) -> GameResult:
        return self.game_result
    
    def get_board(self) -> Board: 
        return self.board

size = 11;
input_dim = (17, size, size)
action_count = size * size + 1

def play_one_game(device: torch.device, inference_model: nn.Module) -> SelfPlayGameBuffer:
    game_buffer = SelfPlayGameBuffer()

    board = FiveInARowBoard((size, size), 16)
    b = board.render();
    network = inference_model
    @torch.no_grad()
    def eval_position(
        state: np.ndarray,
    ) -> Tuple[Iterable[np.ndarray], Iterable[float]]:
        """Give a game state tensor, returns the action probabilities
        and estimated state value from current player's perspective."""
        state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state).to(dtype=torch.float32, device=device, non_blocking=True)

        pi_logits, v = network(state)

        pi_logits = torch.detach(pi_logits)
        v = torch.detach(v)
        # print(f"NN v: {v}, {pi_logits}")

        pi = torch.softmax(pi_logits, dim=-1).cpu().numpy()
        v = v.cpu().numpy()

        B, *_ = state.shape

        v = np.squeeze(v, axis=1)
        v = v.tolist()  # To list

        return pi, v
    root = MCTSNode(action_count, eval_position, board, 1);
    step_count = 0
    for i in range(0, 13 * 13):
        sim_count = 400;
        print(f"Start sim {sim_count}")
        start_time = time.time()
        if step_count <= 2:
            root.add_noise(0.06)
        else:
            root.add_noise(0.03)
        print(f"Step {step_count}")
        while sim_count > 0:
            node, count = root.expand_until_leaf_or_terminal(sim_count, 1)
            sim_count -= count
            node.back_update()
        end_time = time.time()

        print(root.children_N[:-1].reshape(size, size))
        print(root.children_Q[:-1].reshape(size, size))
        print(root.formula[:-1].reshape(size, size))
        print(root.policy[:-1].reshape(size, size))
        print(f"To play {root.get_board().get_current_player()}, V: {root.get_v()}")
        game_buffer.add_sample(root.get_board().get_stack(), root.get_training_pi(1.0), root.get_board().get_current_player())
        next_node = root.commit_next_move()
        b = next_node.get_board().render();
        root = next_node
        print(f"===<commited one move> time {end_time - start_time}=====")
        print(b)
        step_count += 1
        res = root.get_result()
        if res is not GameResult.UNDECIDED:
            print(f"winner: {root.get_board().get_winner()}")
            game_buffer.finalize_game(root.get_board().get_winner())

            break
    return game_buffer

def compute_losses(network, state, target_pi, target_v) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_pi_logits, pred_v = network(state)

    # Policy cross-entropy loss
    policy_loss = F.cross_entropy(pred_pi_logits, target_pi, reduction='mean')

    # State value MSE loss
    value_loss = F.mse_loss(pred_v.squeeze(), target_v.squeeze(), reduction='mean')

    return policy_loss, value_loss


def train_one_batch(replay_buffer: ReplayBuffer, model: nn.Module, lr_scheduler: torch.optim.lr_scheduler.MultiStepLR, 
          optimizer: torch.optim.Optimizer,
          device: torch.device):
    states, policies, values = replay_buffer.sample_batch()
    
    states = torch.from_numpy(states).float().to(device)         # (B, C, H, W)
    policies = torch.from_numpy(policies).float().to(device)     # (B, A)
    values = torch.from_numpy(values).float().to(device)         # (B, 1)
    optimizer.zero_grad()

    pi_loss, v_loss = compute_losses(model, states, policies, values)
    loss = pi_loss + 0.5 * v_loss
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    stats = {
        'policy_loss': pi_loss.detach().item(),
        'value_loss': v_loss.detach().item(),
        "total_loss": loss.detach().item(),
        'learning_rate': lr_scheduler.get_last_lr()[0],
        'total_samples': len(replay_buffer),
    }
    print(f"{stats}")

def train_one_epoch(replay_buffer: ReplayBuffer, model: nn.Module, lr_scheduler: torch.optim.lr_scheduler.MultiStepLR, 
          optimizer: torch.optim.Optimizer,
          device: torch.device):
    for i in range(0, len(replay_buffer) // replay_buffer.batch_size):
        train_one_batch(replay_buffer, model, lr_scheduler, optimizer, device) 


def generate_replays(
    replay_buffer: ReplayBuffer,
    infer_model: AlphaZeroNet, 
    max_replace_sample_ratio: float,
    device: torch.device
) -> ReplayBuffer:
    replace_sample_count = 0
    total_games = 0
    print(f"Starting replay generation: buffer size={len(replay_buffer)}, max_samples={replay_buffer.max_samples}, max_replace_ratio={max_replace_sample_ratio}")
    while (not replay_buffer.is_full()) or (replace_sample_count < max_replace_sample_ratio * replay_buffer.max_samples):
        game = play_one_game(device, infer_model)
        added = replay_buffer.add_game(game)
        replace_sample_count += added
        total_games += 1
        print(f"Game {total_games} finished. Added {added} samples. Replaced Sample Count: {replace_sample_count}, Buffer size: {len(replay_buffer)} / {replay_buffer.max_samples}")
    print(f"Replay generation complete. Total games played: {total_games}, total samples replaced/added: {replace_sample_count}")
    return replay_buffer

def load_latest_model(model_manager: ModelCheckpointManager, device: torch.device) -> AlphaZeroNet:
    infer_model = AlphaZeroNet(input_dim, action_count).to(device)
    weights = model_manager.load_latest(device)
    if weights is not None:
        print("loading weights")
        infer_model.load_state_dict(weights)
    return infer_model

def train_on_latest_model_epoch(
    replay_buffer: ReplayBuffer, 
    model_manager: ModelCheckpointManager, 
    model: AlphaZeroNet,
    device: torch.device, 
    epoch: int,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    optimizer: torch.optim.Optimizer
) -> ReplayBuffer:
    for i in range(0, epoch):
        train_one_epoch(replay_buffer, model, lr_scheduler, optimizer, device)
    return replay_buffer

def generate_replays_and_train(
    replay_buffer: ReplayBuffer, 
    model_manager: ModelCheckpointManager, 
    model: AlphaZeroNet,
    device: torch.device, 
    epoch: int,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    optimizer: torch.optim.Optimizer,
    tournament_games: int = 20,
    tournament_sims: int = 100,
    early_stop_lead: int = 5,
    dump_dir: str = "dump"
) -> ReplayBuffer:
    iteration = 0
    
    while True:
        iteration += 1
        print(f"\n🔄 Training Iteration {iteration}")
        print("=" * 50)

        # # Load the latest model from the model manager
        infer_model = load_latest_model(model_manager, device)
        
        # Generate replays using the current model
        replay_buffer = generate_replays(
            replay_buffer=replay_buffer,
            infer_model=infer_model,
            max_replace_sample_ratio=0.75,
            device=device
        )
        training_model = load_latest_model(model_manager, device)

        train_on_latest_model_epoch(replay_buffer, model_manager, training_model, device, epoch, lr_scheduler, optimizer)
        model_manager.save(training_model)
        
        # After training, run tournament between current model (index 0) and previous model (index 1)
        print(f"\n🏆 Running tournament to evaluate new model...")
        
        # Import battle functions
        from battle import run_tournament_and_dump_loser
        
        try:
            tournament_result = run_tournament_and_dump_loser(
                model_manager=model_manager,
                model1_index=0,  # Current model (latest)
                model2_index=1,  # Previous model
                num_games=tournament_games,
                board_size=(11, 11),
                sim_count=tournament_sims,
                temperature=1.0,
                add_noise=True,
                device=device,
                dump_dir=dump_dir,
                early_stop_lead=early_stop_lead
            )
            
            # Check if the current model (index 0) lost
            if tournament_result['dump_info']['winner_index'] == 1:  # Previous model won
                print(f"💀 Current model (index 0) lost to previous model (index 1)")
                print(f"🗑️  Current model was moved to {dump_dir}")
                print(f"🔄 Continuing training with previous model...")
            elif tournament_result['dump_info']['winner_index'] == 0:  # Current model won
                print(f"✅ Current model (index 0) won against previous model (index 1)")
                print(f"🗑️  Previous model was moved to {dump_dir}")
            else:
                print(f"🤝 Tournament ended in a tie - both models moved to {dump_dir}")
                
        except Exception as e:
            print(f"⚠️  Tournament failed: {e}")
            print(f"🔄 Continuing training without tournament evaluation...")
        

def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    np.set_printoptions(linewidth=sys.maxsize) 

    input_dim = (17, size, size)

    # device = torch.device("cpu")
    if torch.backends.mps.is_available():
        print("Init for mac")
        device = torch.device("mps")
    else:
        print("User cpu")
        device = torch.device("cpu")
    
    model_manager = ModelCheckpointManager(type(AlphaZeroNet), "/Users/sjin2/PPP/AlphaKindaZero/after-fix")

    replay_buffer = ReplayBuffer(600, 32)
    
    network = load_latest_model(model_manager, device)
    # weights = model_manager.load_latest(device)
    # if weights is not None:
    #     print("loading weights")
    #     network.load_state_dict(weights)
    # optimizer = torch.optim.SGD(
    #     network.parameters(),
    #     lr=1e-4,
    #     momentum=0.9,
    #     weight_decay=1e-4,
    # )
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4, weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000, 200000], gamma=0.1)
    
    # Start the training loop with tournament evaluation
    print("Start training with tournament evaluation!!!!!!")
    generate_replays_and_train(
        replay_buffer=replay_buffer,
        model_manager=model_manager,
        model=network,
        device=device,
        epoch=30,  # Train for 1 epoch per iteration
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        tournament_games=20,
        tournament_sims=100,
        early_stop_lead=5,
        dump_dir="/Users/sjin2/PPP/AlphaKindaZero/after-fix-dump"
    ) 

    # size = 11;
    # input_dim = (17, size, size)
    # board = FiveInARowBoard((size, size), 16)
    # b = board.render();
    # action_count = size * size + 1
    # network = AlphaZeroNet(input_dim, action_count).to(device)
    # @torch.no_grad()
    # def eval_position(
    #     state: np.ndarray,
    # ) -> Tuple[Iterable[np.ndarray], Iterable[float]]:
    #     """Give a game state tensor, returns the action probabilities
    #     and estimated state value from current player's perspective."""
    #     state = np.expand_dims(state, axis=0)
    #     state = torch.from_numpy(state).to(dtype=torch.float32, device=device, non_blocking=True)

    #     pi_logits, v = network(state)

    #     pi_logits = torch.detach(pi_logits)
    #     v = torch.detach(v)

    #     pi = torch.softmax(pi_logits, dim=-1).cpu().numpy()
    #     v = v.cpu().numpy()

    #     B, *_ = state.shape

    #     v = np.squeeze(v, axis=1)
    #     v = v.tolist()  # To list


    #     # pi = pi[0]
    #     # v = v[0]


    #     return pi, v
    # root = MCTSNode(action_count, eval_position, board, 1);

    # for i in range(0, 13 * 13):
    #     sim_count = 200;
    #     start_time = time.time()

    #     while sim_count > 0:
    #         node, count = root.expand_until_leaf_or_terminal(sim_count, 1)
    #         sim_count -= count
    #         node.back_update()
    #     end_time = time.time()

    #     print(root.children_N[:-1].reshape(size, size))
    #     print(root.children_Q[:-1].reshape(size, size))
    #     next_node = root.commit_next_move()
    #     b = next_node.get_board().render();
    #     root = next_node
    #     print(f"===<commited one move> time {end_time - start_time}=====")
    #     print(b)

    #     res = root.get_result()
    #     if res is not GameResult.UNDECIDED:
    #         print(f"winner: {root.get_board().get_winner()}")
    #         break



if __name__ == '__main__':
    main()
