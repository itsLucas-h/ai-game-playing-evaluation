import sys
import os
import time
import numpy as np
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.q_learning_agent import QLearningAgent
from agents.score import score_position_q_learning
from agents.minimax_agent import MinimaxAgent
from plots.plot import plot_rewards
from config import (
    Q_LEARNING_ALPHA,
    Q_LEARNING_GAMMA,
    Q_LEARNING_EPSILON,
    Q_LEARNING_EPSILON_DECAY,
    Q_LEARNING_EPSILON_MIN,
    TRAINING_EPISODES,
    MINIMAX_DEPTH,
)
from game.game import (
    create_board,
    get_next_open_row,
    drop_piece,
    winning_move,
    is_draw,
    is_valid_location,
)

# Create directory for training data if it doesn't exist
if not os.path.exists("training"):
    os.makedirs("training")


def train_q_learning_vs_minimax():
    # Print the training configuration for reference
    print("\n----- Training Configuration -----")
    print(f"Alpha (Learning Rate): {Q_LEARNING_ALPHA}")
    print(f"Gamma (Discount Factor): {Q_LEARNING_GAMMA}")
    print(f"Epsilon (Exploration Rate): {Q_LEARNING_EPSILON}")
    print(f"Epsilon Decay Rate: {Q_LEARNING_EPSILON_DECAY}")
    print(f"Minimum Epsilon: {Q_LEARNING_EPSILON_MIN}")
    print(f"Training Episodes: {TRAINING_EPISODES}")
    print(f"Minimax Depth: {MINIMAX_DEPTH}")

    # Initialize Q-Learning and Minimax agents with respective parameters
    q_agent = QLearningAgent(
        alpha=Q_LEARNING_ALPHA,
        gamma=Q_LEARNING_GAMMA,
        epsilon=Q_LEARNING_EPSILON,
        epsilon_decay=Q_LEARNING_EPSILON_DECAY,
        epsilon_min=Q_LEARNING_EPSILON_MIN,
    )
    minimax_agent = MinimaxAgent(depth=MINIMAX_DEPTH)
    rewards = []  # Track rewards per episode
    reward_window = deque(maxlen=100)  # Store last 100 rewards for moving average
    q_agent_penalty_log = []  # Track penalties incurred by Q-Learning agent
    q_agent_wins = 0  # Counter for Q-learning agent wins

    # Initialize metrics for performance tracking of both agents
    total_minimax_decision_time = 0
    total_nodes_explored = 0
    total_nodes_pruned = 0
    minimax_moves = 0

    total_q_moves = 0
    total_q_penalties = 0
    total_reward_sum = 0

    print("\n----- Q-Learning Agent Training against Minimax Start -----")

    # Training loop for each episode
    for episode in range(1, TRAINING_EPISODES + 1):
        board = create_board()  # Initialize game board
        game_over = False
        total_reward = 0  # Reward for current episode
        q_agent_penalties = 0  # Penalties for current episode
        turn = 0  # 0 for Q-learning agent's turn, 1 for Minimax's

        while not game_over:
            if turn == 0:  # Q-learning agent's turn
                state = q_agent.get_state(board)  # Get current state of board
                action = q_agent.choose_action(board, train=True)  # Select action
                if action is None:  # If no valid action, apply penalty and end game
                    game_over = True
                    reward = -10
                    total_reward += reward
                    q_agent_penalties += 1
                    break
                row = get_next_open_row(board, action)  # Get row for the action
                drop_piece(board, row, action, 1)  # Execute action

                # Calculate reward for the action
                reward = score_position_q_learning(board, 1)
                if winning_move(board, 1):  # Win condition
                    reward += 50
                    game_over = True
                    q_agent_wins += 1
                elif is_draw(board):  # Draw condition
                    reward -= 10
                    game_over = True
                else:
                    q_agent_penalties += (
                        reward if reward < 0 else 0
                    )  # Count penalty if reward is negative

                total_q_moves += 1
                total_reward_sum += reward
                total_q_penalties += abs(reward) if reward < 0 else 0

                next_state = q_agent.get_state(board)  # Get next state
                q_agent.learn(state, action, reward, next_state)  # Update Q-table
                total_reward += reward  # Accumulate reward for episode

            else:  # Minimax agent's turn
                # Measure Minimax decision time and performance
                start_time = time.time()
                action, nodes_explored, nodes_pruned = minimax_agent.get_move(
                    board, 2, track_nodes=True
                )
                decision_time = time.time() - start_time

                total_minimax_decision_time += decision_time
                total_nodes_explored += nodes_explored
                total_nodes_pruned += nodes_pruned
                minimax_moves += 1

                # Execute Minimax action if valid
                if action is not None and is_valid_location(board, action):
                    row = get_next_open_row(board, action)
                    drop_piece(board, row, action, 2)
                    if winning_move(board, 2):  # Minimax agent win condition
                        total_reward -= 50
                        game_over = True

            turn = 1 - turn  # Alternate turns between Q-learning and Minimax agents

        # Update epsilon for Q-learning agent to reduce exploration over time
        q_agent.update_epsilon()
        rewards.append(total_reward)  # Track reward for this episode
        reward_window.append(total_reward)  # Update reward window
        q_agent_penalty_log.append(q_agent_penalties)  # Log penalties for episode

        # Display metrics every 100 episodes
        if episode % 100 == 0:
            avg_reward = np.mean(reward_window)
            avg_q_agent_penalties = np.mean(q_agent_penalty_log[-100:])
            win_rate = (q_agent_wins / 100) * 100
            q_agent_wins = 0  # Reset win count for next interval

            print(
                f"Ep: {episode:>5} | Total Reward: {total_reward:>8.2f} | ε: {q_agent.epsilon:>8.4f} "
                f"| Avg Reward: {avg_reward:>8.2f} | Win Rate: {win_rate:>6.2f}% "
                f"| Avg Penalties: {avg_q_agent_penalties:>6.2f}"
            )

    print("----- Q-Learning Agent Training against Minimax Complete -----\n")
    plot_rewards(rewards, context="minimax")  # Plot reward trends over episodes

    # Calculate and print summary statistics
    total_episodes = len(rewards)
    final_win_rate = (q_agent_wins / total_episodes) * 100
    avg_reward_per_move = total_reward_sum / total_q_moves if total_q_moves else 0
    avg_penalty_per_move = total_q_penalties / total_q_moves if total_q_moves else 0
    avg_decision_time = (
        total_minimax_decision_time / minimax_moves if minimax_moves else 0
    )
    avg_nodes_explored = total_nodes_explored / minimax_moves if minimax_moves else 0
    avg_nodes_pruned = total_nodes_pruned / minimax_moves if minimax_moves else 0

    # Print final statistics summary
    print("\n----- End of Training Summary -----")
    print("Q-learning Agent Performance:")
    print(f"Final Win Rate: {final_win_rate:.2f}%")
    print(f"Average Reward per Move: {avg_reward_per_move:.2f}")
    print(f"Average Penalty per Move: {avg_penalty_per_move:.2f}")
    print("\nMinimax Agent Performance:")
    print(f"Average Decision Time per Move: {avg_decision_time:.4f} sec")
    print(f"Average Nodes Explored per Move: {avg_nodes_explored:.2f}")
    print(f"Average Nodes Pruned per Move: {avg_nodes_pruned:.2f}")
    print("-----------------------------------")

    # Save the trained Q-table for the Q-learning agent
    q_agent.save_q_table_qvm("training/q_table(qvm).npy")
    return q_agent
