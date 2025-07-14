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
        self.black_board = np.zeros(size, dtype=np.float32)
        self.white_board = np.zeros(size, dtype=np.float32)
        self.to_play = 1  # Player 1 goes first
        self.queue.push_board(self.current_board)
        self.queue.set_player(self.to_play == 1)

    def current_binary_board(self) -> np.ndarray:
        if self.to_play == 1:
            return self.black_board
        elif self.to_play == 2:
            return self.white_board

    def play_step(self, row: int, col: int) -> bool:
        """Attempt to play a move for the current player (1 or 2)."""
        if self.current_board[row, col] != 0:
            return False  # Invalid move
        self.current_board[row, col] = self.to_play
        binary_board = self.current_binary_board()
        binary_board[row, col] = 1
        # Push the plane for the player who just moved **before** we flip `to_play`
        self.queue.push_board(binary_board.copy())
        # Now switch to the next player and update the indicator mask
        self.to_play = 3 - self.to_play  # 1 ↔ 2
        self.queue.set_player(self.to_play == 1)
        return True
    
    def pass_move(self):
        # Store current plane first, then flip player and update mask
        self.queue.push_board(self.current_binary_board().copy())
        self.to_play = 3 - self.to_play  # 1 ↔ 2
        self.queue.set_player(self.to_play == 1)
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
    
    def render_stack(self) -> str:
        """
        Renders the whole stack with the following formatting:
        - Black planes: "*"
        - White planes: "o" 
        - Player planes: "B" for player 1, "W" for player 2
        """
        stack = self.get_stack()
        height, width = self.size
        num_planes = stack.shape[0]
        
        output = []
        output.append(f"Stack shape: {stack.shape}")
        output.append("=" * (width * 2 + 10))
        
        for plane_idx in range(num_planes):
            plane = stack[plane_idx]
            output.append(f"Plane {plane_idx}:")
            to_play_odd_plane = (num_planes - 2) % 2
            # Determine what this plane represents
            if plane_idx < num_planes - 1:  # History planes
                if (plane_idx % 2 == to_play_odd_plane) ^ (self.to_play == 1):  # Black planes
                    plane_type = "Black"
                    symbol = "*"
                else:  # White planes
                    plane_type = "White" 
                    symbol = "o"
            else:  # Player indicator plane
                plane_type = "Player"
                symbol = "B" if self.to_play == 1 else "W"
            
            output.append(f"  Type: {plane_type}")
            
            # Render the plane
            for row in range(height):
                row_str = "  "
                for col in range(width):
                    if plane[row, col] == 1:
                        row_str += symbol + " "
                    else:
                        row_str += ". "
                output.append(row_str)
            output.append("")
        
        return "\n".join(output)
    
    def flatten_pos(self, pos: Tuple[int, int]) -> int:
        return np.ravel_multi_index(pos, self.get_current_board().shape)
    
    def unflatten_index(self, index: int) -> Tuple[int, int]:
        return np.unravel_index(index, self.get_current_board().shape)
