from __future__ import annotations

import copy
from enum import Enum, auto
from typing import Callable, Iterable, Mapping, Optional, Tuple
import numpy as np
import torch

from alphago_nn import AlphaZeroNet
from board import Board, GameResult
from five_in_a_row_board import FiveInARowBoard



class MCTSNode:
    def __init__(self, children_size: int, policy_func: Callable[[np.ndarray], Tuple[Iterable[np.ndarray], Iterable[float]]], 
                 board: Board, root_to_play: int, parent: MCTSNode = None):
        self.children_W = np.zeros(children_size, dtype=np.float32)
        self.children_N = np.zeros(children_size, dtype=np.float32)
        self.children_Q = np.zeros(children_size, dtype=np.float32)
        self.children_index: Mapping[int, MCTSNode] = {}
        self.parent = parent
        self.root_to_play = root_to_play
        self.to_play = board.get_current_player()
        self.policy_func = policy_func
        self.board = board
        self.children_size = children_size
        self.game_result = board.get_game_result(root_to_play);
        self.policy: Optional[np.ndarray] = None
        self.v: Optional[float] = None
    
    def pick_next_move(self) -> int:
        return np.argmax(self.children_N)
    
    def commit_next_move(self) -> MCTSNode:
        action = self.pick_next_move()
        row, col = self.board.unflatten_index(action)
        new_board = copy.deepcopy(self.board)
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
        input = self.board.get_stack()
        
        if self.v is None:
            policy_arr, v_arr = self.policy_func(input)
            v = v_arr[0];
            policy = policy_arr[0];
            self.v = v
            self.policy = policy
        else:
            v = self.v
            policy = self.policy
        
        N = np.sum(self.children_N)

        formula = self.children_Q + c * policy / (1 + self.children_N)
        legal_actions = self.board.get_legal_moves().flatten();
        legal_actions = np.append(legal_actions, False);
        illegal_actions = np.logical_not(legal_actions)
        formula[illegal_actions] = -1000
        # print(f"{formula}\n")

        action = np.argmax(formula)
        row, col = self.board.unflatten_index(action)

        new_board = copy.deepcopy(self.board)
        new_board.play_step(row, col)
        leaf_node = False
        if action in self.children_index:
            new_node = self.children_index[action]
        else:
            new_node = MCTSNode(self.children_size, self.policy_func, new_board, self.root_to_play, self)
            self.children_index[action] = new_node
            leaf_node = True

        tmp = self
        while tmp is not None:
            tmp.children_N[action] += 1
            if tmp.to_play == self.root_to_play:
                print(f"+{v}\n")
                tmp.children_W[action] += v
            else:
                print(f"-{v}\n")
                tmp.children_W[action] -= v
            tmp.children_Q[action] = tmp.children_W[action] / tmp.children_N[action]
            tmp = tmp.parent

        return (new_node, leaf_node)
    
    def get_result(self) -> GameResult:
        return self.game_result
    
    def get_board(self) -> Board: 
        return self.board
    

def main():
    size = 11;
    input_dim = (17, size, size)
    board = FiveInARowBoard((size, size), 16)
    b = board.render();
    action_count = size * size + 1
    network = AlphaZeroNet(input_dim, action_count)
    @torch.no_grad()
    def eval_position(
        state: np.ndarray,
    ) -> Tuple[Iterable[np.ndarray], Iterable[float]]:
        """Give a game state tensor, returns the action probabilities
        and estimated state value from current player's perspective."""
        state = np.expand_dims(state, axis=0)

        state = torch.from_numpy(state).to(dtype=torch.float32)

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

    for i in range(0, 5):
        sim_count = 0;
        while sim_count < 1600:
            tmp = root
            while sim_count < 1600:
                child, leaf = tmp.expand(1)
                tmp = child
                sim_count += 1
                res = child.get_result()
                if leaf or res is not GameResult.UNDECIDED:
                    b = child.get_board().render();
                    print(b)
                    print(res)
                    break
        print(root.children_N[:-1].reshape(size, size))
        print(root.children_Q[:-1].reshape(size, size))
        next_node = root.commit_next_move()
        b = next_node.get_board().render();
        root = next_node
        print(b)



if __name__ == '__main__':
    main()
