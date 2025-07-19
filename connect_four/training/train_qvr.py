import os
import numpy as np
from collections import deque
from agents.q_learning_agent import QLearningAgent
from plots.plot import plot_rewards
from agents.score import score_position_q_learning
from agents.random_agent import RandomAgent
from config import (
    Q_LEARNING_ALPHA,
    Q_LEARNING_GAMMA,
    Q_LEARNING_EPSILON,
    Q_LEARNING_EPSILON_DECAY,
    Q_LEARNING_EPSILON_MIN,
    TRAINING_EPISODES,
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


def train_q_learning_vs_random():
    # Print the training configuration parameters
    print("\n----- Training Configuration -----")
    print(f"Alpha (Learning Rate): {Q_LEARNING_ALPHA}")
    print(f"Gamma (Discount Factor): {Q_LEARNING_GAMMA}")
    print(f"Epsilon (Exploration Rate): {Q_LEARNING_EPSILON}")
    print(f"Epsilon Decay Rate: {Q_LEARNING_EPSILON_DECAY}")
    print(f"Minimum Epsilon: {Q_LEARNING_EPSILON_MIN}")
    print(f"Training Episodes: {TRAINING_EPISODES}")

    # Initialize Q-Learning and Random agents with their respective parameters
    q_agent = QLearningAgent(
        alpha=Q_LEARNING_ALPHA,
        gamma=Q_LEARNING_GAMMA,
        epsilon=Q_LEARNING_EPSILON,
        epsilon_decay=Q_LEARNING_EPSILON_DECAY,
        epsilon_min=Q_LEARNING_EPSILON_MIN,
    )
    random_agent = RandomAgent()
    rewards = []  # Track rewards per episode
    reward_window = deque(maxlen=100)  # Store last 100 rewards for moving average
    q_agent_penalty_log = []  # Track penalties incurred by Q-Learning agent
    q_agent_wins = 0  # Counter for Q-learning agent wins

    # Initialize metrics for Q-learning agent's performance
    total_q_moves = 0
    total_q_penalties = 0
    total_reward_sum = 0

    print("\n----- Q-Learning Agent Training against Random Agent Start -----")

    # Training loop for each episode
    for episode in range(1, TRAINING_EPISODES + 1):
        board = create_board()  # Initialize game board
        game_over = False
        total_reward = 0  # Reward for current episode
        q_agent_penalties = 0  # Penalties for current episode
        turn = 0  # 0 for Q-learning agent's turn, 1 for Random agent's

        while not game_over:
            if turn == 0:  # Q-learning agent's turn
                state = q_agent.get_state(board)  # Get current state of board
                action = q_agent.choose_action(board, train=True)  # Select action
                if action is None:  # If no valid action, apply penalty and end game
                    game_over = True
                    reward = -10  # Penalty for invalid action
                    total_reward += reward
                    q_agent_penalties += abs(reward)
                    total_q_penalties += abs(reward)
                    break

                row = get_next_open_row(board, action)  # Get row for the action
                drop_piece(board, row, action, 1)  # Execute action

                # Calculate reward for the action
                reward = score_position_q_learning(board, 1)
                if winning_move(board, 1):  # Win condition
                    reward += 50
                    game_over = True
                    q_agent_wins += 1  # Increment win counter
                elif is_draw(board):  # Draw condition
                    reward -= 10  # Penalty for a draw
                    game_over = True
                elif reward < 0:
                    q_agent_penalties += abs(
                        reward
                    )  # Count penalty if reward is negative
                    total_q_penalties += abs(reward)

                next_state = q_agent.get_state(board)  # Get next state
                q_agent.learn(state, action, reward, next_state)  # Update Q-table
                total_reward += reward  # Accumulate reward for episode

                # Track moves, rewards, and penalties
                total_q_moves += 1
                total_reward_sum += reward

            else:  # Random agent's turn
                # Random agent selects a valid move
                action = random_agent.get_move(board)
                if action is not None and is_valid_location(board, action):
                    row = get_next_open_row(board, action)
                    drop_piece(board, row, action, 2)

                    # Check if Random agent wins
                    if winning_move(board, 2):
                        total_reward -= 50
                        game_over = True

            turn = 1 - turn  # Switch turns between Q-learning and Random agents

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

    print("----- Q-Learning Agent Training Complete -----\n")
    plot_rewards(rewards, context="random")  # Plot reward trends over episodes

    # End-of-training summary
    total_episodes = len(rewards)
    final_win_rate = (
        sum([1 for r in rewards if r >= 50]) / total_episodes
    ) * 100  # Overall win rate based on wins

    # Calculating final average reward and penalty per move
    avg_reward_per_move = total_reward_sum / total_q_moves if total_q_moves else 0
    avg_penalty_per_move = total_q_penalties / total_q_moves if total_q_moves else 0

    # Display final statistics summary
    print("\n----- End of Training Summary -----")
    print("Q-learning Agent Performance:")
    print(f"Final Win Rate: {final_win_rate:.2f}%")
    print(f"Average Reward per Move: {avg_reward_per_move:.2f}")
    print(f"Average Penalty per Move: {avg_penalty_per_move:.2f}")
    print("-----------------------------------")

    # Save the trained Q-table for the Q-learning agent
    q_agent.save_q_table_qvr("training/q_table(qvr).npy")

    return q_agent
