from enum import Enum, auto
from io import StringIO
import os
import sys
from typing import Tuple, Optional

import numpy as np

from matrix_queue import MatrixQueue


class GameResult(Enum):
    UNDECIDED = auto()
    WIN = auto()
    LOSE = auto()
    DRAW = auto()

class Board:
    def __init__(self, size: Tuple[int, int], num_stack: int = 8):
        self.size = size
        self.num_stack = num_stack
        self.queue = MatrixQueue(num_stack, size)
        self.current_board = np.zeros(size, dtype=np.float32)
        self.to_play = 1  # Player 1 goes first
        self.queue.push_board(self.current_board)
        self.queue.set_player(self.to_play)

    def play_step(self, row: int, col: int) -> bool:
        """Attempt to play a move for the current player (1 or 2)."""
        if self.current_board[row, col] != 0:
            return False  # Invalid move
        self.current_board[row, col] = self.to_play
        self.to_play = 3 - self.to_play  # Switch between 1 and 2
        self.queue.push_board(self.current_board.copy())
        self.queue.set_player(self.to_play)
        return True
    
    def pass_move(self):
        self.to_play = 3 - self.to_play  # Switch between 1 and 2
        self.queue.push_board(self.current_board.copy())
        self.queue.set_player(self.to_play)
        return True

    def get_stack(self) -> np.ndarray:
        """Returns the full stack (history + player indicator)."""
        return self.queue.get_stack()

    def get_current_board(self) -> np.ndarray:
        return self.current_board

    def get_current_player(self) -> int:
        return self.to_play  # 1 or 2
    
    def get_the_other_player(self) -> int:
        return 3 - self.to_play

    def get_winner(self) -> Optional[int]:
        """Stub for winner logic. Override for specific game rules."""
        return None
    
    def is_draw(self) -> bool:
        """Stub: Should return True if the game is a draw. Override in subclasses."""
        return False
    
    def get_legal_moves(self) -> np.ndarray:
        return self.current_board
    
    def get_game_result(self, player: Optional[int] = None) -> GameResult:
        if player is None:
            player = self.get_current_player()

        winner = self.get_winner()
        if winner is None:
            if self.is_draw():
                return GameResult.DRAW
            return GameResult.UNDECIDED
        return GameResult.WIN if winner == player else GameResult.LOSE
    
    def render(self):
        """
        Renders a NumPy boolean matrix as ASCII art, using '●' for True
        and ' ' for False.
        """
        matrix = self.get_current_board()
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Input must be a NumPy boolean array.")
        output = []
        for row in matrix:
            output.append(" ".join(["*" if val == 1 else ("O" if val == 2 else "-") for val in row]))
        return "\n".join(output)
    
    def flatten_pos(self, pos: Tuple[int, int]) -> int:
        return np.ravel_multi_index(pos, self.get_current_board().shape)
    
    def unflatten_index(self, index: int) -> Tuple[int, int]:
        return np.unravel_index(index, self.get_current_board().shape)
