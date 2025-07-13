from typing import List, Optional, Tuple
import numpy as np
import random

class TrainingSample:
    def __init__(self, state: np.ndarray, policy: np.ndarray, to_play: int):
        self.state = state                 # (C, H, W) numpy array
        self.policy = policy              # (num_actions,) softmax probabilities from MCTS
        self.to_play = to_play            # 1 or 2
        self.result: Optional[float] = None  # final game outcome from this player's perspective

    def set_result(self, z: float):
        """Set final outcome from this player's perspective (+1 win, -1 loss, 0 draw)"""
        self.result = z


class SelfPlayGameBuffer:
    def __init__(self):
        self.samples: List[TrainingSample] = []
        self.final_result: Optional[int] = None  # +1 = win for player 1, -1 = win for player 2, 0 = draw

    def add_sample(self, state: np.ndarray, policy: np.ndarray, to_play: int):
        """Add a state, MCTS policy, and player to play"""
        self.samples.append(TrainingSample(state, policy, to_play))

    def finalize_game(self, winner: int):
        """Call this at game end to assign result to all samples"""
        self.final_result = winner
        for sample in self.samples:
            if winner == 0 or winner is None:
                sample.set_result(0)
            elif sample.to_play == winner:
                print(f"set +1 for sample {winner}, {sample.to_play}")
                sample.set_result(+1)
            else:
                print(f"set -1 for sample {winner}, {sample.to_play}")
                sample.set_result(-1)

    def get_training_data(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Return list of (state, policy, value) tuples for training"""
        return [
            (sample.state, sample.policy, sample.result)
            for sample in self.samples if sample.result is not None
        ]

    def __len__(self):
        return len(self.samples)



class ReplayBuffer:
    def __init__(self, max_samples: int, batch_size: int):
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []

    def add_game(self, game_buffer: SelfPlayGameBuffer):
        """Add all samples from one finished game"""
        samples = game_buffer.get_training_data()
        self.buffer.extend(samples)

        # FIFO eviction
        excess = len(self.buffer) - self.max_samples
        if excess > 0:
            self.buffer = self.buffer[excess:]

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomly sample a training batch of (state, pi, z)"""
        batch = random.sample(self.buffer, k=min(self.batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)

        states_np = np.stack(states).astype(np.float32)        # (B, C, H, W)
        policies_np = np.stack(policies).astype(np.float32)    # (B, num_actions)
        values_np = np.array(values).astype(np.float32)[:, None]  # (B, 1)

        return states_np, policies_np, values_np

    def __len__(self):
        return len(self.buffer)
