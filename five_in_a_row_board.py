from typing import Optional, Tuple

import numpy as np
from board import Board


class FiveInARowBoard(Board):
    def __init__(self, size: Tuple[int, int] = (15, 15), num_stack: int = 8):
        super().__init__(size=size, num_stack=num_stack)

    def get_winner(self) -> Optional[int]:
        board = self.get_current_board()
        for player in (1, 2):
            # Horizontal
            if self._check_lines(board, player, axis=1):
                return player
            # Vertical
            if self._check_lines(board, player, axis=0):
                return player
            # Diagonal
            if self._check_diagonals(board, player):
                return player
        return None

    def _check_lines(self, board: np.ndarray, player: int, axis: int) -> bool:
        """Check rows or columns (axis=1 for rows, 0 for columns)."""
        if axis == 1:  # Horizontal
            lines = board
        else:  # Vertical
            lines = board.T

        for line in lines:
            if self._has_five_in_a_row(line, player):
                return True
        return False

    def _check_diagonals(self, board: np.ndarray, player: int) -> bool:
        h, w = board.shape
        for offset in range(-h + 5, w - 4):
            diag1 = np.diagonal(board, offset=offset)
            diag2 = np.diagonal(np.fliplr(board), offset=offset)
            if self._has_five_in_a_row(diag1, player) or self._has_five_in_a_row(diag2, player):
                return True
        return False

    def _has_five_in_a_row(self, line: np.ndarray, player: int) -> bool:
        """Check if there are 5 continuous stones of `player` in the line."""
        count = 0
        for value in line:
            count = count + 1 if value == player else 0
            if count >= 5:
                return True
        return False
    
    def is_draw(self) -> bool:
        """Game is a draw if the board is full and no winner."""
        board = self.get_current_board()
        return np.all(board != 0) and self.get_winner() is None
    
    def get_legal_moves(self) -> np.ndarray:
        """
        Returns a boolean matrix of shape (height, width) where True indicates a legal move.
        """
        board = self.get_current_board()
        return board == 0