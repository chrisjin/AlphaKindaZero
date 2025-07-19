import numpy as np
from typing import Tuple, Optional

class MatrixQueue:
    def __init__(self, num_stack: int, size: Tuple[int, int]):
        """
        :param num_stack: Number of historical board states to store
        :param size: Size of each board (rows, cols)
        """
        self.num_stack = num_stack
        self.size = size  # (height, width)

        # Stack layout:
        #   0 .. num_stack-1 : Historical board states (oldest -> latest)
        #   -3              : Latest board state (same as index num_stack)
        #   -2              : Last-move indicator plane (1 at last move position, else 0)
        #   -1              : Player-to-move indicator plane (all 0 for player 2, all 1 for player 1)
        # That is, we store `num_stack` board planes + 1 last-move plane + 1 player-indicator plane.
        self.stack = np.zeros((num_stack + 2, *size), dtype=np.float32)

    def push_board(self, matrix: np.ndarray, last_move: Optional[Tuple[int, int]] = None):
        """Push a new board state and (optionally) update the last-move plane.

        Args:
            matrix: 2-D NumPy array representing the latest board.
            last_move: Optional (row, col) coordinate of the move just played.
        """
        assert matrix.shape == self.size, f"Expected shape {self.size}, got {matrix.shape}"
        # Shift board states only (exclude last-move and player planes)
        #   Example with num_stack = 8 (total planes = 10):
        #   Indices 0-7   -> board history
        #   Index  8 (-2) -> last move plane (keep)
        #   Index  9 (-1) -> player plane     (keep)
        #   We want to drop plane 0, shift 1..7 -> 0..6, and write new board to 7.
        if self.num_stack > 1:
            self.stack[:-3] = self.stack[1:-2]
        # Store the newest board just before the two special planes
        self.stack[-3] = matrix

        # Update last-move plane if provided
        self.stack[-2].fill(0)
        if last_move is not None:
            self.set_last_move(last_move)

    def set_player(self, to_play: int):
        self.stack[-1][:] = to_play  # fill the mask with 0s or 1s

    def get_stack(self) -> np.ndarray:
        """Returns the full stack: N boards + 1 player-indicator plane."""
        return self.stack

    def get_last_board(self) -> np.ndarray:
        """Returns the most recent board state (not player mask)."""
        return self.stack[-3]

    def get_player_plane(self) -> np.ndarray:
        """Returns the plane indicating who is to play."""
        return self.stack[-1]

    def set_last_move(self, last_move: Tuple[int, int]):
        """Set the last-move plane to a single 1 at the provided position.

        Args:
            last_move: (row, col) of the move. All other cells are set to 0.
        """
        row, col = last_move
        assert 0 <= row < self.size[0] and 0 <= col < self.size[1], "Last move out of board bounds"
        self.stack[-2][row, col] = 1

    def get_last_move_plane(self) -> np.ndarray:
        """Returns the last-move indicator plane (1 where the last move was played)."""
        return self.stack[-2]
    
    def extra_plane_count(self) -> int:
        return 2

    def total_plane_count(self) -> int:
        return self.num_stack + 2