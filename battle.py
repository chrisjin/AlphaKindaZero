import torch
import numpy as np
import time
import os
from typing import Optional, Tuple, Iterable

from alphago_nn import AlphaZeroNet
from board import Board, GameResult
from five_in_a_row_board import FiveInARowBoard
from mcts import MCTSNode
from model_manager import ModelCheckpointManager


def run_simulations_and_make_move(
    root: MCTSNode, 
    sim_count: int, 
    temperature: float = 1.0,
    add_noise: bool = False
) -> Tuple[MCTSNode, int]:
    """
    Run MCTS simulations and make a move using existing MCTSNode methods.
    
    Args:
        root: Root node of the MCTS tree
        sim_count: Number of simulations to run
        temperature: Temperature for action selection (1.0 = proportional to visits, 0.0 = greedy)
        add_noise: Whether to add Dirichlet noise to the root policy
    
    Returns:
        Tuple of (new_root_node, action_taken)
    """
    if add_noise:
        root.add_noise()
    
    # Run simulations using existing expand_until_leaf_or_terminal and back_update
    while sim_count > 0:
        node, count = root.expand_until_leaf_or_terminal(sim_count, 1)
        sim_count -= count
        node.back_update()
    
    # Select action based on temperature
    if temperature == 0.0:
        # Greedy selection - use existing pick_next_move
        action = root.pick_next_move()
    else:
        # Temperature-based selection
        pi = root.get_training_pi(temperature)
        action = np.random.choice(len(pi), p=pi)
    
    # Make the move using existing commit_next_move
    new_root = root.commit_next_move()
    
    return new_root, action


def load_model_by_index(model_manager: ModelCheckpointManager, index: int, input_dim: Tuple[int, int, int], action_count: int, device: torch.device) -> AlphaZeroNet:
    """Load a trained model from checkpoint by index."""
    model = AlphaZeroNet(input_dim, action_count).to(device)
    
    weights = model_manager.load_by_index(index, device)
    if weights is not None:
        model.load_state_dict(weights)
        print(f"✅ Loaded model index {index}")
    else:
        print(f"⚠️  No model found at index {index}, using random weights")
    
    model.eval()
    return model


def create_eval_function(model: AlphaZeroNet, device: torch.device):
    @torch.no_grad()
    def eval_position(
        state: np.ndarray,
    ) -> Tuple[Iterable[np.ndarray], Iterable[float]]:
        """Give a game state tensor, returns the action probabilities
        and estimated state value from current player's perspective."""
        state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state).to(dtype=torch.float32, device=device, non_blocking=True)

        pi_logits, v = model(state)

        pi_logits = torch.detach(pi_logits)
        v = torch.detach(v)
        # print(f"NN v: {v}, {pi_logits}")

        pi = torch.softmax(pi_logits, dim=-1).cpu().numpy()
        v = v.cpu().numpy()

        B, *_ = state.shape

        v = np.squeeze(v, axis=1)
        v = v.tolist()  # To list

        return pi, v
    
    return eval_position


def battle_models(
    model_manager: ModelCheckpointManager,
    model1_index: int = 0,  # Latest model (Black)
    model2_index: int = 1,  # Second to last model (White)
    input_dim: Tuple[int, int, int] = (17, 11, 11),
    sim_count: int = 400,
    temperature: float = 1.0,
    add_noise: bool = False,
    device: torch.device = None
) -> dict:
    """
    Let two models battle against each other.
    
    Args:
        model_manager: ModelCheckpointManager instance
        model1_index: Index of first model (plays as Black, 0 = latest)
        model2_index: Index of second model (plays as White, 1 = second to last)
        board_size: Size of the board
        sim_count: Number of MCTS simulations per move
        temperature: Temperature for action selection
        add_noise: Whether to add Dirichlet noise
        device: Device to run on
    
    Returns:
        Dictionary with game results
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    # Setup
    board_size = (input_dim[1], input_dim[2])
    action_count = input_dim[1] * input_dim[2] + 1
    
    # Show available models
    print("📁 Available models:")
    model_manager.print_available_models()
    
    # Load models by index
    model1 = load_model_by_index(model_manager, model1_index, input_dim, action_count, device)
    model2 = load_model_by_index(model_manager, model2_index, input_dim, action_count, device)
    
    # Create evaluation functions
    eval_func1 = create_eval_function(model1, device)
    eval_func2 = create_eval_function(model2, device)
    
    # Initialize game
    board = FiveInARowBoard(board_size, input_dim[0] - 1)
    root = MCTSNode(action_count, eval_func1, board, 1)
    
    move_count = 0
    max_moves = board_size[0] * board_size[1]
    
    print(f"🎮 Battle started: Model {model1_index} (Black) vs Model {model2_index} (White)")
    print(f"📊 Settings: {sim_count} sims, temp={temperature}, noise={add_noise}")
    print(f"📋 Board: {board_size[0]}x{board_size[1]} (input_dim: {input_dim})")
    print("-" * 50)
    
    # Game loop
    while move_count < max_moves:
        current_player = root.get_board().get_current_player()
        current_model = model1 if current_player == 1 else model2
        current_eval = eval_func1 if current_player == 1 else eval_func2
        
        # Update root's evaluation function for current player
        root.policy_func = current_eval
        
        # print(f"Move {move_count + 1}: Player {current_player} ({'Black' if current_player == 1 else 'White'})")
        
        # Make move
        start_time = time.time()
        root, action = run_simulations_and_make_move(
            root, sim_count, temperature, add_noise
        )
        move_time = time.time() - start_time
        
        # Get move coordinates
        row, col = root.get_board().unflatten_index(action)
        
        # print(f"  Action: ({row}, {col}) in {move_time:.2f}s")
        # print(f"  Board state:")
        # print(root.get_board().render())
        # print()
        
        # Check for game end
        winner = root.get_board().get_winner()
        if winner is not None:
            print(f"🏆 Game Over! Winner: Player {winner} ({'Black' if winner == 1 else 'White'})")
            break
        
        move_count += 1
    
    # Final result
    final_winner = root.get_board().get_winner()
    if final_winner is None and move_count >= max_moves:
        print("🤝 Game ended in a draw (board full)")
        final_winner = 0
    
    result = {
        'winner': final_winner,
        'moves': move_count,
        'model1_index': model1_index,
        'model2_index': model2_index,
        'settings': {
            'sim_count': sim_count,
            'temperature': temperature,
            'add_noise': add_noise,
            'input_dim': input_dim,
            'board_size': board_size
        }
    }
    
    print("-" * 50)
    print(f"📈 Final Result: {result}")
    
    return result


def run_tournament(
    model_manager: ModelCheckpointManager,
    model1_index: int = 0,
    model2_index: int = 1,
    num_games: int = 10,
    input_dim: Tuple[int, int, int] = (17, 11, 11),
    sim_count: int = 200,
    temperature: float = 1.0,
    add_noise: bool = False,
    device: torch.device = None
) -> dict:
    """
    Run a tournament between two models, alternating who starts as black/white.
    
    Args:
        model_manager: ModelCheckpointManager instance
        model1_index: Index of first model
        model2_index: Index of second model
        num_games: Number of games to play
        board_size: Size of the board
        sim_count: Number of MCTS simulations per move
        temperature: Temperature for action selection
        add_noise: Whether to add Dirichlet noise
        device: Device to run on
    
    Returns:
        Dictionary with tournament results
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    board_size = (input_dim[1], input_dim[2])
    print(f"🏆 Tournament: Model {model1_index} vs Model {model2_index}")
    print(f"📊 {num_games} games, {sim_count} sims per move, temp={temperature}")
    print(f"📋 Board: {board_size[0]}x{board_size[1]} (input_dim: {input_dim})")
    print("=" * 60)
    
    results = []
    model1_wins = 0
    model2_wins = 0
    draws = 0
    
    for game in range(num_games):
        # Alternate between model1 and model2 for black/white
        if game % 2 == 0:
            # Even games: model1 as black, model2 as white
            black_model_index = model1_index
            white_model_index = model2_index
            print(f"\n🎮 Game {game + 1}/{num_games}: Model {model1_index} (Black) vs Model {model2_index} (White)")
        else:
            # Odd games: model2 as black, model1 as white
            black_model_index = model2_index
            white_model_index = model1_index
            print(f"\n🎮 Game {game + 1}/{num_games}: Model {model2_index} (Black) vs Model {model1_index} (White)")
        
        # Run the battle
        result = battle_models(
            model_manager=model_manager,
            model1_index=black_model_index,
            model2_index=white_model_index,
            input_dim=input_dim,
            sim_count=sim_count,
            temperature=temperature,
            add_noise=add_noise,
            device=device
        )
        
        # Track results
        winner = result['winner']
        if winner == 1:  # Black won
            if black_model_index == model1_index:
                model1_wins += 1
                print(f"✅ Model {model1_index} wins!")
            else:
                model2_wins += 1
                print(f"✅ Model {model2_index} wins!")
        elif winner == 2:  # White won
            if white_model_index == model1_index:
                model1_wins += 1
                print(f"✅ Model {model1_index} wins!")
            else:
                model2_wins += 1
                print(f"✅ Model {model2_index} wins!")
        else:  # Draw
            draws += 1
            print(f"🤝 Draw!")
        
        # Display running score
        print(f"📊 Current Score: Model {model1_index} {model1_wins} - {model2_wins} Model {model2_index} (Draws: {draws})")
        
        results.append(result)
    
    # Tournament summary
    print("\n" + "=" * 60)
    print(f"🏆 TOURNAMENT RESULTS")
    print(f"Model {model1_index} vs Model {model2_index}")
    print(f"Model {model1_index} wins: {model1_wins}")
    print(f"Model {model2_index} wins: {model2_wins}")
    print(f"Draws: {draws}")
    print(f"Model {model1_index} win rate: {model1_wins/num_games*100:.1f}%")
    print(f"Model {model2_index} win rate: {model2_wins/num_games*100:.1f}%")
    print(f"Final Score: Model {model1_index} {model1_wins} - {model2_wins} Model {model2_index}")
    
    tournament_result = {
        'model1_index': model1_index,
        'model2_index': model2_index,
        'model1_wins': model1_wins,
        'model2_wins': model2_wins,
        'draws': draws,
        'total_games': num_games,
        'model1_win_rate': model1_wins / num_games,
        'model2_win_rate': model2_wins / num_games,
        'individual_results': results
    }
    
    return tournament_result


def run_tournament_with_early_stop(
    model_manager: ModelCheckpointManager,
    model1_index: int = 0,
    model2_index: int = 1,
    num_games: int = 10,
    input_dim: Tuple[int, int, int] = (17, 11, 11),
    sim_count: int = 200,
    temperature: float = 1.0,
    add_noise: bool = True,
    device: torch.device = None,
    early_stop_lead: int = 5
) -> dict:
    """
    Run a tournament between two models with early stopping if one model gets too far ahead.
    
    Args:
        model_manager: ModelCheckpointManager instance
        model1_index: Index of first model
        model2_index: Index of second model
        num_games: Number of games to play
        board_size: Size of the board
        sim_count: Number of MCTS simulations per move
        temperature: Temperature for action selection
        add_noise: Whether to add Dirichlet noise
        device: Device to run on
        early_stop_lead: Stop tournament if one model is this many games ahead
    
    Returns:
        Dictionary with tournament results
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    print(f"🏆 Tournament: Model {model1_index} vs Model {model2_index}")
    print(f"📊 {num_games} games, {sim_count} sims per move, temp={temperature}")
    print(f"⏹️  Early stop if lead >= {early_stop_lead}")
    print("=" * 60)
    
    results = []
    model1_wins = 0
    model2_wins = 0
    draws = 0
    
    for game in range(num_games):
        # Check for early stopping
        if abs(model1_wins - model2_wins) >= early_stop_lead:
            print(f"\n⏹️  EARLY STOP: Model {model1_index if model1_wins > model2_wins else model2_index} is {abs(model1_wins - model2_wins)} games ahead")
            break
        
        # Alternate between model1 and model2 for black/white
        if game % 2 == 0:
            # Even games: model1 as black, model2 as white
            black_model_index = model1_index
            white_model_index = model2_index
            print(f"\n🎮 Game {game + 1}/{num_games}: Model {model1_index} (Black) vs Model {model2_index} (White)")
        else:
            # Odd games: model2 as black, model1 as white
            black_model_index = model2_index
            white_model_index = model1_index
            print(f"\n🎮 Game {game + 1}/{num_games}: Model {model2_index} (Black) vs Model {model1_index} (White)")
        
        # Run the battle
        result = battle_models(
            model_manager=model_manager,
            model1_index=black_model_index,
            model2_index=white_model_index,
            input_dim=input_dim,
            sim_count=sim_count,
            temperature=temperature,
            add_noise=add_noise,
            device=device
        )
        
        # Track results
        winner = result['winner']
        if winner == 1:  # Black won
            if black_model_index == model1_index:
                model1_wins += 1
                print(f"✅ Model {model1_index} wins!")
            else:
                model2_wins += 1
                print(f"✅ Model {model2_index} wins!")
        elif winner == 2:  # White won
            if white_model_index == model1_index:
                model1_wins += 1
                print(f"✅ Model {model1_index} wins!")
            else:
                model2_wins += 1
                print(f"✅ Model {model2_index} wins!")
        else:  # Draw
            draws += 1
            print(f"🤝 Draw!")
        
        # Display running score
        print(f"📊 Current Score: Model {model1_index} {model1_wins} - {model2_wins} Model {model2_index} (Draws: {draws})")
        
        results.append(result)
    
    # Tournament summary
    print("\n" + "=" * 60)
    print(f"🏆 TOURNAMENT RESULTS")
    print(f"Model {model1_index} vs Model {model2_index}")
    print(f"Model {model1_index} wins: {model1_wins}")
    print(f"Model {model2_index} wins: {model2_wins}")
    print(f"Draws: {draws}")
    print(f"Games played: {len(results)}")
    if len(results) < num_games:
        print(f"⏹️  Tournament stopped early due to {early_stop_lead}-game lead")
    print(f"Model {model1_index} win rate: {model1_wins/len(results)*100:.1f}%")
    print(f"Model {model2_index} win rate: {model2_wins/len(results)*100:.1f}%")
    print(f"Final Score: Model {model1_index} {model1_wins} - {model2_wins} Model {model2_index}")
    
    tournament_result = {
        'model1_index': model1_index,
        'model2_index': model2_index,
        'model1_wins': model1_wins,
        'model2_wins': model2_wins,
        'draws': draws,
        'total_games': len(results),
        'planned_games': num_games,
        'early_stopped': len(results) < num_games,
        'model1_win_rate': model1_wins / len(results) if len(results) > 0 else 0,
        'model2_win_rate': model2_wins / len(results) if len(results) > 0 else 0,
        'individual_results': results
    }
    
    return tournament_result


def run_comprehensive_tournament(
    model_manager: ModelCheckpointManager,
    model_indices: list = None,
    games_per_match: int = 4,
    input_dim: Tuple[int, int, int] = (17, 11, 11),
    sim_count: int = 50,
    temperature: float = 1.0,
    add_noise: bool = False,
    device: torch.device = None
) -> dict:
    """
    Run a comprehensive tournament between all models in the provided list.
    Each model plays against every other model.
    
    Args:
        model_manager: ModelCheckpointManager instance
        model_indices: List of model indices to include in tournament
        games_per_match: Number of games per model pair
        board_size: Size of the board
        sim_count: Number of MCTS simulations per move
        temperature: Temperature for action selection
        add_noise: Whether to add Dirichlet noise
        device: Device to run on
    
    Returns:
        Dictionary with comprehensive tournament results
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    # Use default range if no indices provided
    if model_indices is None:
        model_indices = list(range(0, 17))  # 0 to 16
    
    # Remove duplicates and sort
    model_indices = sorted(list(set(model_indices)))
    
    num_models = len(model_indices)
    total_matches = num_models * (num_models - 1) // 2
    total_games = total_matches * games_per_match
    
    print(f"🏆 COMPREHENSIVE TOURNAMENT")
    print(f"📊 Models: {model_indices} ({num_models} models)")
    print(f"📊 {total_matches} matches, {total_games} total games")
    print(f"📊 {games_per_match} games per match, {sim_count} sims per move")
    print("=" * 80)
    
    # Initialize scores for each model
    model_scores = {i: {'wins': 0, 'losses': 0, 'draws': 0, 'total_games': 0} for i in model_indices}
    
    match_count = 0
    
    # Run tournaments between all pairs
    for i_idx, i in enumerate(model_indices):
        for j in model_indices[i_idx + 1:]:
            match_count += 1
            print(f"\n🎯 Match {match_count}/{total_matches}: Model {i} vs Model {j}")
            
            # Run tournament between these two models
            result = run_tournament(
                model_manager=model_manager,
                model1_index=i,
                model2_index=j,
                num_games=games_per_match,
                input_dim=input_dim,
                sim_count=sim_count,
                temperature=temperature,
                add_noise=add_noise,
                device=device
            )
            
            # Update scores
            model_scores[i]['wins'] += result['model1_wins']
            model_scores[i]['losses'] += result['model2_wins']
            model_scores[i]['draws'] += result['draws']
            model_scores[i]['total_games'] += result['total_games']
            
            model_scores[j]['wins'] += result['model2_wins']
            model_scores[j]['losses'] += result['model1_wins']
            model_scores[j]['draws'] += result['draws']
            model_scores[j]['total_games'] += result['total_games']
            
            # Display current rankings
            print(f"\n📊 CURRENT RANKINGS:")
            display_rankings(model_scores)
    
    # Final rankings
    print("\n" + "=" * 80)
    print(f"🏆 FINAL TOURNAMENT RESULTS")
    print(f"📊 Models: {model_indices}")
    display_rankings(model_scores)
    
    return {
        'model_scores': model_scores,
        'model_indices': model_indices,
        'total_matches': total_matches,
        'total_games': total_games,
        'games_per_match': games_per_match
    }


def display_rankings(model_scores: dict):
    """Display current rankings of all models."""
    # Calculate win rates and sort by win rate
    rankings = []
    for model_index, scores in model_scores.items():
        if scores['total_games'] > 0:
            win_rate = scores['wins'] / scores['total_games']
            rankings.append((model_index, scores['wins'], scores['losses'], scores['draws'], scores['total_games'], win_rate))
    
    # Sort by win rate (descending), then by total wins (descending)
    rankings.sort(key=lambda x: (x[5], x[1]), reverse=True)
    
    print(f"{'Rank':<4} {'Model':<6} {'Wins':<6} {'Losses':<8} {'Draws':<6} {'Games':<6} {'Win Rate':<10}")
    print("-" * 50)
    
    for rank, (model_index, wins, losses, draws, total_games, win_rate) in enumerate(rankings, 1):
        print(f"{rank:<4} {model_index:<6} {wins:<6} {losses:<8} {draws:<6} {total_games:<6} {win_rate*100:.1f}%")


def run_tournament_and_dump_loser(
    model_manager: ModelCheckpointManager,
    model1_index: int = 0,
    model2_index: int = 1,
    num_games: int = 20,
    input_dim: Tuple[int, int, int] = (17, 11, 11),
    sim_count: int = 100,
    temperature: float = 1.0,
    add_noise: bool = True,
    device: torch.device = None,
    dump_dir: str = "dump",
    early_stop_lead: int = 5
) -> dict:
    """
    Run a tournament between two models and move the losing model to a dump directory.
    
    Args:
        model_manager: ModelCheckpointManager instance
        model1_index: Index of first model
        model2_index: Index of second model
        num_games: Number of games to play
        board_size: Size of the board
        sim_count: Number of MCTS simulations per move
        temperature: Temperature for action selection
        add_noise: Whether to add Dirichlet noise
        device: Device to run on
        dump_dir: Directory to move the losing model to
    
    Returns:
        Dictionary with tournament results and dump information
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    print(f"🏆 TOURNAMENT WITH DUMP: Model {model1_index} vs Model {model2_index}")
    print(f"📊 {num_games} games, {sim_count} sims per move, temp={temperature}")
    print(f"🗑️  Loser will be moved to: {dump_dir}")
    print(f"⏹️  Early stop if lead >= {early_stop_lead}")
    print("=" * 60)
    
    # Run the tournament with early stopping
    tournament_result = run_tournament_with_early_stop(
        model_manager=model_manager,
        model1_index=model1_index,
        model2_index=model2_index,
        num_games=num_games,
        input_dim=input_dim,
        sim_count=sim_count,
        temperature=temperature,
        add_noise=add_noise,
        device=device,
        early_stop_lead=early_stop_lead
    )
    
    # Determine winner and loser
    model1_wins = tournament_result['model1_wins']
    model2_wins = tournament_result['model2_wins']
    
    print(f"\n🏆 TOURNAMENT COMPLETE")
    print(f"Model {model1_index} wins: {model1_wins}")
    print(f"Model {model2_index} wins: {model2_wins}")
    print(f"Draws: {tournament_result['draws']}")
    
    # Determine winner and handle dumping based on model age
    newer_model_index = min(model1_index, model2_index)  # Lower index = newer model
    older_model_index = max(model1_index, model2_index)  # Higher index = older model
    
    if model1_wins > model2_wins:
        winner_index = model1_index
        loser_index = model2_index
        print(f"🏆 Winner: Model {winner_index}")
        print(f"💀 Loser: Model {loser_index}")
    elif model2_wins > model1_wins:
        winner_index = model2_index
        loser_index = model1_index
    else:
        # Tie - keep both models
        winner_index = None
        loser_index = None
    
    # Handle dumping based on model age
    if winner_index is not None:
        if winner_index == newer_model_index:
            # Newer model won - keep both models
            print(f"✅ Newer model {newer_model_index} won against older model {older_model_index}")
            print(f"📦 Keeping both models (newer model is better)")
        else:
            # Older model won - dump the newer model
            print(f"💀 Newer model {newer_model_index} lost to older model {older_model_index}")
            print(f"🗑️  Dumping newer model {newer_model_index} (older model is better)")
            
            # Create dump directory
            os.makedirs(dump_dir, exist_ok=True)
            
            # Move the newer (losing) model to dump
            success = model_manager.move_model_by_index(newer_model_index, dump_dir, move_file=True)
            
            if success:
                print(f"🗑️  Moved newer model {newer_model_index} to {dump_dir}")
            else:
                print(f"❌ Failed to move newer model {newer_model_index}")
    
    elif winner_index is None:
        # Tie case - keep both models
        print(f"🤝 Tournament ended in a tie")
        print(f"📦 Keeping both models (no clear winner)")

    # Add dump information to result
    tournament_result['dump_info'] = {
        'winner_index': winner_index,
        'loser_index': loser_index,
        'dump_directory': dump_dir,
        'tie': winner_index is None
    }
    
    return tournament_result


def main():
    """Example usage of the battle system."""
    # Create model manager
    # model_manager = ModelCheckpointManager(type(AlphaZeroNet), 
    #     "/Users/sjin2/PPP/AlphaKindaZero/after-fix")  
    # model_manager = ModelCheckpointManager(type(AlphaZeroNet), 
    #     "/Users/sjin2/PPP/AlphaKindaZero/8by8-le")
    model_manager = ModelCheckpointManager(type(AlphaZeroNet), 
        "/Users/sjin2/PPP/AlphaKindaZero/8by8-le-aug")
    # Run comprehensive tournament for the latest 3 models
    # input_dim = (17, 11, 11)
    input_dim = (5, 8, 8)

    tournament_result = run_comprehensive_tournament(
        model_manager=model_manager,
        model_indices=[0, 250],  # Latest 3 models
        games_per_match=20,  # 10 games per model pair
        input_dim=input_dim,
        sim_count=100,   # 100 sims per move
        temperature=1.0,
        add_noise=True,  # Add noise for evaluation
        device=None  # Auto-detect
    )
    
    print(f"\n🎯 Comprehensive tournament completed!")
    print(f"Tournament involved {len(tournament_result['model_indices'])} models: {tournament_result['model_indices']}")
    print(f"Total matches: {tournament_result['total_matches']}")
    print(f"Total games: {tournament_result['total_games']}")
    
    # Find the best model
    best_model = max(tournament_result['model_scores'].items(), key=lambda x: x[1]['wins'])[0]
    print(f"🏆 Best model: Model {best_model}")


if __name__ == "__main__":
    main() 