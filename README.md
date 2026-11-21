# AlphaKindaZero

An experimental AlphaZero-style Gomoku engine built with PyTorch. It combines Monte Carlo tree search, a residual policy/value network, self-play, and checkpoint tournaments.

## Features

- PUCT-style tree search with neural policy priors and value estimates.
- Gomoku rules, legal-move masking, and stacked board-state inputs.
- Self-play data generation and a replay buffer for policy/value training.
- Timestamped checkpoints, paired matches, and round-robin evaluation.

## Setup

The Python code requires NumPy and PyTorch. From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy torch
```

Model weights are not included. Training starts with a newly initialized network when the configured checkpoint directory is empty.

## Training

Configuration is currently defined in the source files. Before starting, change the checkpoint directory in `mcts.py`'s `main()` and its `dump_dir` argument to local experiment directories.

```sh
python mcts.py
```

The training loop generates self-play samples, updates the network, saves a checkpoint, and evaluates it against the previous checkpoint. It continues until interrupted. Losing checkpoints are moved to the dump directory; a tied evaluation can move both checkpoints there.

The current defaults are:

| Setting | Value | Location |
| --- | --- | --- |
| Board size | 8 × 8 | `mcts.py`: `size` |
| Network input | `(5, 8, 8)` | `mcts.py`: `input_dim` |
| Action slots | 65, including an unused pass slot | `mcts.py`: `action_count` |
| Self-play search budget | 100 simulations per move | `mcts.py`: `play_one_game()` |
| Replay capacity / batch size | 2,000 / 32 | `mcts.py`: `main()` |
| Training epochs per iteration | 15 | `mcts.py`: `main()` |
| Evaluation games / search budget | Up to 20 games / 100 simulations per move | `mcts.py`: `main()` |

The entry point uses Apple MPS when available and CPU otherwise.

## Checkpoint evaluation

`battle.py` supports paired matches and round-robin tournaments. Before running its example, set the checkpoint directory, input dimensions, and checkpoint indices in `main()`:

```sh
python battle.py
```

The checked-in evaluation example uses input shape `(17, 11, 11)` and indices `range(0, 4)`. For checkpoints produced by the current training defaults, use `(5, 8, 8)` and select indices that exist. Checkpoints are indexed by file modification time, with index 0 referring to the newest file. Board dimensions, input channels, and network architecture must match the saved weights.

## Code layout

| File | Purpose |
| --- | --- |
| [`mcts.py`](mcts.py) | Search, self-play, policy/value losses, and the training loop |
| [`alphago_nn.py`](alphago_nn.py) | Residual network with policy and value heads |
| [`board.py`](board.py), [`five_in_a_row_board.py`](five_in_a_row_board.py) | Board state, history, legal moves, wins, and draws |
| [`training_sample_queue.py`](training_sample_queue.py), [`matrix_queue.py`](matrix_queue.py) | Replay samples and stacked state storage |
| [`model_manager.py`](model_manager.py) | Checkpoint saving, loading, and indexing |
| [`battle.py`](battle.py) | Model matches and tournament evaluation |
| [`issues.md`](issues.md) | Experiment logs and debugging notes |
| [`scratch.py`](scratch.py) | Model-output comparison experiment using an older input shape |

## Experiment notes

The results below are recorded in [`issues.md`](issues.md). Their checkpoint files are not included in the repository.

- A round-robin tournament of 17 checkpoints, with 64 games per checkpoint, produced win rates of 39.1%–56.2%. The newest checkpoint scored 48.4%. This run did not show a stable improvement from training.
- A later note records a 5–0 result against checkpoint `20250712_190041` after increasing the simulation budget from 200 to 400 and further training. The short match and incomplete configuration record do not isolate the effect of either change.
- The history includes a [Dirichlet-noise correction](https://github.com/chrisjin/AlphaKindaZero/commit/14d3404) and a subsequent [learning-rate adjustment](https://github.com/chrisjin/AlphaKindaZero/commit/132cbbb). The notes record that insufficient exploration noise had previously been treated as a learning-rate problem.
- Other fixes address [optimizer parameters bound to the wrong model and a missing `train()` call](https://github.com/chrisjin/AlphaKindaZero/commit/cb125f9), as well as [player-plane encoding, tie-breaking, and draws](https://github.com/chrisjin/AlphaKindaZero/commit/b86c185).

### Known limitations

Experiment paths and settings are hard-coded, and the training and evaluation examples use different model shapes. Historical results should be read with their original configuration and bug history in mind.

### Development directions

- Compare legal-action-only exploration noise with the current full-action-space sampling.
- Save configuration and random seeds alongside checkpoints.
- Evaluate against fixed baselines with more games and alternating colors.
- Add checks for state encoding, training modes, optimizer parameters, and draw targets.

## Background and acknowledgments

Gomoku provides a small environment for exploring neural-network-guided search, with automated theorem proving as a longer-term motivation.

The policy/value network in `alphago_nn.py` is adapted from Michael Hu's *The Art of Reinforcement Learning*. The file retains the original copyright and MIT notice. See [Michael Hu's AlphaZero implementation](https://github.com/michaelnny/alpha_zero) for the upstream project and license.
