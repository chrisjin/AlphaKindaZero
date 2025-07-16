from typing import List, Optional, Tuple
import numpy as np
import random

class TrainingSample:
    def __init__(self, state: np.ndarray, policy: np.ndarray, to_play: int):
        self.state = state                 # (C, H, W) numpy array
        self.policy = policy              # (num_actions,) softmax probabilities from MCTS
        self.to_play = to_play            # 1 or 2
        self.result: Optional[float] = None  # final game outcome from this player's perspective

    def __init__(self, state: np.ndarray, policy: np.ndarray, to_play: int, result: Optional[float] = None):
        self.state = state                 # (C, H, W) numpy array
        self.policy = policy              # (num_actions,) softmax probabilities from MCTS
        self.to_play = to_play            # 1 or 2
        self.result = result              # final game outcome from this player's perspective

    def set_result(self, z: float):
        """Set final outcome from this player's perspective (+1 win, -1 loss, 0 draw)"""
        self.result = z
    
    def extract_policy(self) -> Tuple[np.ndarray, float]:
        """Extract the policy from the state"""
        state_dim = self.state.shape[1:]
        policy_matrix = self.policy[:-1].reshape(state_dim)
        pass_move = self.policy[-1]
        return policy_matrix, pass_move
    
    def reshape_back(self, policy_matrix: np.ndarray, pass_move: float) -> np.ndarray:
        """Reshape the policy matrix back to the original shape"""
        return np.concatenate([policy_matrix.flatten(), [pass_move]])

    def policy_rot90(self, k: int, axis: Tuple[int, int] = (0, 1)) -> np.ndarray:
        policy_matrix, pass_move = self.extract_policy()
        rotated_policy_matrix = np.rot90(policy_matrix, k=k, axes=axis)
        return self.reshape_back(rotated_policy_matrix, pass_move)
    
    def policy_flip(self, axis: int) -> np.ndarray:
        policy_matrix, pass_move = self.extract_policy()
        flipped_policy_matrix = np.flip(policy_matrix, axis=axis)
        return self.reshape_back(flipped_policy_matrix, pass_move)


class SelfPlayGameBuffer:
    def __init__(self):
        self.samples: List[TrainingSample] = []
        self.final_result: Optional[int] = None  # +1 = win for player 1, -1 = win for player 2, 0 = draw

    def add_sample(self, state: np.ndarray, policy: np.ndarray, to_play: int):
        """Add a state, MCTS policy, and player to play"""
        self.samples.append(TrainingSample(state, policy, to_play))

    def finalize_game(self, winner: int, data_augmentation: bool = False):
        """Call this at game end to assign result to all samples, with early steps less impactful."""
        self.final_result = winner
        total_steps = len(self.samples)
        for idx, sample in enumerate(self.samples):
            # Current step: idx (0-based), so currentstep = idx
            # Impact factor: 1 / sqrt(total_steps - idx)
            impact = 1.0
            # if total_steps - idx > 0:
            #     impact = 1.0 / np.sqrt(total_steps - idx)
            if winner == 0 or winner is None:
                sample.set_result(0)
            elif sample.to_play == winner:
                value = +1 * impact
                print(f"set {value:.3f} for sample {winner}, {sample.to_play} (step {idx}, impact {impact:.3f})")
                sample.set_result(value)
            else:
                value = -1 * impact
                print(f"set {value:.3f} for sample {winner}, {sample.to_play} (step {idx}, impact {impact:.3f})")
                sample.set_result(value)
        if data_augmentation:
            self.augment_data()
    
    def augment_data(self):
        """Augment the data by flipping the board and swapping the player."""
        new_samples = []
        for sample in self.samples:
            rotated_state = np.rot90(sample.state, k=1, axes=(1, 2))
            rotated_policy = sample.policy_rot90(1)
            new_samples.append(TrainingSample(rotated_state, rotated_policy, sample.to_play, sample.result))
            rotated_state = np.rot90(sample.state, k=2, axes=(1, 2))
            rotated_policy = sample.policy_rot90(2)
            new_samples.append(TrainingSample(rotated_state, rotated_policy, sample.to_play, sample.result))
            rotated_state = np.rot90(sample.state, k=3, axes=(1, 2))
            rotated_policy = sample.policy_rot90(3)
            new_samples.append(TrainingSample(rotated_state, rotated_policy, sample.to_play, sample.result))
            
            rotated_state = np.flip(sample.state, axis=1)
            rotated_policy = sample.policy_flip(0)
            new_samples.append(TrainingSample(rotated_state, rotated_policy, sample.to_play, sample.result))
            rotated_state = np.flip(sample.state, axis=2)      
            rotated_policy = sample.policy_flip(1)
            new_samples.append(TrainingSample(rotated_state, rotated_policy, sample.to_play, sample.result))
            rotated_state = np.flip(sample.state, axis=(1, 2))
            rotated_policy = sample.policy_flip((0, 1))
            new_samples.append(TrainingSample(rotated_state, rotated_policy, sample.to_play, sample.result))
        self.samples.extend(new_samples)
    
        
        

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
        self.full = False
        self.replaced_samples = 0

    def add_game(self, game_buffer: SelfPlayGameBuffer) -> int:
        """Add all samples from one finished game"""
        samples = game_buffer.get_training_data()
        self.buffer.extend(samples)

        # FIFO eviction
        excess = len(self.buffer) - self.max_samples
        if excess > 0:
            self.full = True
            self.buffer = self.buffer[excess:]
            self.replaced_samples += excess
        else:
            self.replaced_samples = len(self.buffer)
        return len(samples)
    
    def is_full(self) -> bool:
        return self.full

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomly sample a training batch of (state, pi, z)"""
        # Weighted sampling: later items have higher probability
        buffer_len = len(self.buffer)
        k = min(self.batch_size, buffer_len)
        # Assign weights increasing with index (later = higher weight)
        weights = np.arange(1, buffer_len + 1, dtype=np.float64)
        weights = weights / weights.sum()
        indices = np.random.choice(buffer_len, size=k, replace=False, p=weights)
        batch = [self.buffer[i] for i in indices]
        states, policies, values = zip(*batch)

        states_np = np.stack(states).astype(np.float32)        # (B, C, H, W)
        policies_np = np.stack(policies).astype(np.float32)    # (B, num_actions)
        values_np = np.array(values).astype(np.float32)[:, None]  # (B, 1)

        return states_np, policies_np, values_np

    def __len__(self):
        return len(self.buffer)
        

