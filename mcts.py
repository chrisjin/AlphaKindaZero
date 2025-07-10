from __future__ import annotations

import copy
from enum import Enum, auto
import time
from typing import Callable, Iterable, Mapping, Optional, Tuple
import numpy as np
import torch

from alphago_nn import AlphaZeroNet
from board import Board, GameResult
from five_in_a_row_board import FiveInARowBoard



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

        self.last_action: Optional[int] = last_action 
        input = self.board.get_stack()
        winner = board.get_winner()
        if winner is not None:
            if winner == board.get_current_player():
                self.v = 1.0
            else:
                self.v = -1.0
            print(f"Got winner: {self.v}, {winner}, {board.get_current_player()}")

        else:
            policy_arr, v_arr = self.policy_func(input)
            v = v_arr[0];
            policy = policy_arr[0];
            self.v = v
            self.policy = policy
    
    def pick_next_move(self) -> int:
        return np.argmax(self.children_N)
    
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

    def expand(self, c: float) -> Tuple[MCTSNode, bool]:
        policy = self.policy

        formula = self.children_Q + c * policy * np.sqrt(self.N) / (1 + self.children_N)
        legal_actions = self.board.get_legal_moves().flatten();
        legal_actions = np.append(legal_actions, False);
        illegal_actions = np.logical_not(legal_actions)
        formula[illegal_actions] = -1000
        # print(f"{formula}\n")

        action = np.argmax(formula)
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
    
    def expand_until_leaf_or_terminal(self, limit: int, c: float) -> Tuple[MCTSNode, int]:
        tmp = self
        steps = 1
        while limit > 0:
            if tmp.get_result() is not GameResult.UNDECIDED:
                return (tmp, steps)
            next, is_new_node = tmp.expand(c)
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
        while tmp_parent is not None:
            tmp_parent.children_N[tmp_child.last_action] += 1
            tmp_parent.N += 1
            if tmp_parent.to_play == self.root_to_play:
                tmp_parent.children_W[tmp_child.last_action] -= v
            else:
                tmp_parent.children_W[tmp_child.last_action] += v
            tmp_parent.children_Q[tmp_child.last_action] = tmp_parent.children_W[tmp_child.last_action] / tmp_parent.children_N[tmp_child.last_action]
            tmp_child = tmp_parent
            tmp_parent = tmp_parent.parent

    def get_result(self) -> GameResult:
        return self.game_result
    
    def get_board(self) -> Board: 
        return self.board
    

def main():
    # device = torch.device("cpu")
    if torch.backends.mps.is_available():
        print("Init for mac")
        device = torch.device("mps")
    else:
        print("User cpu")
        device = torch.device("cpu")
    size = 11;
    input_dim = (17, size, size)
    board = FiveInARowBoard((size, size), 16)
    b = board.render();
    action_count = size * size + 1
    network = AlphaZeroNet(input_dim, action_count).to(device)
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

        pi = torch.softmax(pi_logits, dim=-1).cpu().numpy()
        v = v.cpu().numpy()

        B, *_ = state.shape

        v = np.squeeze(v, axis=1)
        v = v.tolist()  # To list


        # pi = pi[0]
        # v = v[0]


        return pi, v
    root = MCTSNode(action_count, eval_position, board, 1);

    for i in range(0, 13 * 13):
        sim_count = 1600;
        while sim_count > 0:
            start_time = time.time()
            node, count = root.expand_until_leaf_or_terminal(sim_count, 1)
            sim_count -= count
            node.back_update()
        end_time = time.time()

        print(root.children_N[:-1].reshape(size, size))
        print(root.children_Q[:-1].reshape(size, size))
        next_node = root.commit_next_move()
        b = next_node.get_board().render();
        root = next_node
        print(f"===<commited one move> time {end_time - start_time}=====")
        print(b)

        res = root.get_result()
        if res is not GameResult.UNDECIDED:
            break



if __name__ == '__main__':
    main()
