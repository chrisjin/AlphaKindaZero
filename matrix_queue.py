import numpy as np
from typing import Tuple

class MatrixQueue:
    def __init__(self, num_stack: int, size: Tuple[int, int]):
        """
        :param num_stack: Number of historical board states to store
        :param size: Size of each board (rows, cols)
        """
        self.num_stack = num_stack
        self.size = size  # (height, width)

        # stack includes num_stack boards + 1 player-indicator plane
        self.stack = np.zeros((num_stack + 1, *size), dtype=np.float32)

    def push_board(self, matrix: np.ndarray):
        """Push a new board state, drop the oldest."""
        assert matrix.shape == self.size, f"Expected shape {self.size}, got {matrix.shape}"
        # Shift board states only (not the last player mask)
        self.stack[:-2] = self.stack[1:-1]
        self.stack[-2] = matrix  # second-to-last is newest board

    def set_player(self, to_play: int):
        self.stack[-1][:] = to_play  # fill the mask with 0s or 1s

    def get_stack(self) -> np.ndarray:
        """Returns the full stack: N boards + 1 player-indicator plane."""
        return self.stack

    def get_last_board(self) -> np.ndarray:
        """Returns the most recent board state (not player mask)."""
        return self.stack[-2]

    def get_player_plane(self) -> np.ndarray:
        """Returns the last plane indicating who is to play."""
        return self.stack[-1]
