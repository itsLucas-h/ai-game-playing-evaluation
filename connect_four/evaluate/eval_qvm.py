import time
from agents.minimax_agent import MinimaxAgent
from agents.q_learning_agent import QLearningAgent
from game.game import create_board, get_next_open_row, drop_piece, winning_move
from agents.score import score_position_q_learning
from config import (
    EVALUATION_GAMES,
    Q_LEARNING_ALPHA,
    Q_LEARNING_GAMMA,
    Q_LEARNING_EPSILON,
    Q_LEARNING_EPSILON_DECAY,
    Q_LEARNING_EPSILON_MIN,
    MINIMAX_DEPTH,
)


def evaluate_q_learning_vs_minimax(q_agent, num_games=EVALUATION_GAMES):
    # Initialize Q-learning and Minimax agents with specified parameters
    q_agent = QLearningAgent(
        alpha=Q_LEARNING_ALPHA,
        gamma=Q_LEARNING_GAMMA,
        epsilon=Q_LEARNING_EPSILON,
        epsilon_decay=Q_LEARNING_EPSILON_DECAY,
        epsilon_min=Q_LEARNING_EPSILON_MIN,
    )
    minimax_agent = MinimaxAgent(depth=MINIMAX_DEPTH)

    # Initialize metrics to track performance during evaluation
    q_wins = minimax_wins = draws = 0
    minimax_moves = 0
    total_decision_time = 0
    total_nodes_explored = total_nodes_pruned = 0
    q_total_reward = 0
    q_total_penalties = 0
    q_total_moves = 0

    # Run evaluation games
    for _ in range(num_games):
        board = create_board()  # Initialize game board
        game_over = False
        current_player = 0  # 0 for Q-learning agent, 1 for Minimax agent

        while not game_over:
            if current_player == 0:  # Q-learning agent's turn
                action = q_agent.choose_action(board, train=False)  # Choose action
                if action is None:  # If no valid action, mark as draw
                    draws += 1
                    break

                row = get_next_open_row(board, action)  # Find row for action
                drop_piece(board, row, action, 1)  # Execute action on board
                reward = score_position_q_learning(board, 1)  # Calculate reward

                # Accumulate reward and move count for Q-learning agent
                q_total_reward += reward
                q_total_moves += 1

                # Track penalties by summing negative rewards
                if reward < 0:
                    q_total_penalties += abs(reward)

                if winning_move(board, 1):  # Check if Q-learning agent wins
                    q_wins += 1
                    game_over = True

            else:  # Minimax agent's turn
                # Measure decision time and track nodes explored and pruned
                start_time = time.time()
                action, explored, pruned = minimax_agent.get_move(
                    board, 2, track_nodes=True
                )
                total_decision_time += time.time() - start_time
                total_nodes_explored += explored
                total_nodes_pruned += pruned
                minimax_moves += 1

                # Execute Minimax agent's move if valid
                if action is not None:
                    row = get_next_open_row(board, action)
                    drop_piece(board, row, action, 2)

                    if winning_move(board, 2):  # Check if Minimax agent wins
                        minimax_wins += 1
                        game_over = True

            current_player = 1 - current_player  # Switch turns between agents

    # Calculate metrics for evaluation summary
    avg_reward_per_move = q_total_reward / q_total_moves if q_total_moves else 0
    avg_penalty_per_game = q_total_penalties / num_games
    avg_decision_time = total_decision_time / minimax_moves if minimax_moves else 0
    avg_nodes_explored = total_nodes_explored / minimax_moves if minimax_moves else 0
    avg_nodes_pruned = total_nodes_pruned / minimax_moves if minimax_moves else 0
    q_win_rate = (q_wins / num_games) * 100
    minimax_win_rate = (minimax_wins / num_games) * 100
    draw_rate = (draws / num_games) * 100
    q_loss_rate = ((num_games - q_wins - draws) / num_games) * 100
    minimax_loss_rate = ((num_games - minimax_wins - draws) / num_games) * 100

    # Print evaluation summary for both agents
    print("\n----- Evaluation Summary -----")
    print("Q-Learning Agent:")
    print(f"Q-learning Agent Wins: {q_wins} ({q_win_rate:.2f}%)")
    print(f"Draws: {draws} ({draw_rate:.2f}%)")
    print(f"Losses: {num_games - q_wins - draws} ({q_loss_rate:.2f}%)")
    print(f"Average Reward per Move: {avg_reward_per_move:.2f}")
    print(f"Average Penalties per Game: {avg_penalty_per_game:.2f}")

    print("\nMinimax Agent:")
    print(f"Minimax Agent Wins: {minimax_wins} ({minimax_win_rate:.2f}%)")
    print(f"Draws: {draws} ({draw_rate:.2f}%)")
    print(f"Losses: {num_games - minimax_wins - draws} ({minimax_loss_rate:.2f}%)")
    print(f"Average Decision Time per Move: {avg_decision_time:.4f} sec")
    print(f"Average Nodes Explored per Move: {avg_nodes_explored:.2f}")
    print(f"Average Nodes Pruned per Move: {avg_nodes_pruned:.2f}\n")
