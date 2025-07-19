import os
import numpy as np
from collections import deque
from agents.q_learning_agent import QLearningAgent
from agents.score import score_position_q_learning
from game.game import (
    create_board,
    get_next_open_row,
    drop_piece,
    winning_move,
    is_draw,
)
from plots.plot import plot_rewards
from config import (
    Q_LEARNING_ALPHA,
    Q_LEARNING_GAMMA,
    Q_LEARNING_EPSILON,
    Q_LEARNING_EPSILON_DECAY,
    Q_LEARNING_EPSILON_MIN,
    TRAINING_EPISODES,
)

if not os.path.exists("training"):
    os.makedirs("training")


def train_q_learning_self_play():
    q_agent = QLearningAgent(
        alpha=Q_LEARNING_ALPHA,
        gamma=Q_LEARNING_GAMMA,
        epsilon=Q_LEARNING_EPSILON,
        epsilon_decay=Q_LEARNING_EPSILON_DECAY,
        epsilon_min=Q_LEARNING_EPSILON_MIN,
    )
    rewards = []
    penalties = []
    reward_window = deque(maxlen=50)
    penalty_window = deque(maxlen=50)

    print("\n----- Q-Learning Self-Play Training Start -----")

    for episode in range(1, TRAINING_EPISODES + 1):
        board = create_board()
        game_over = False
        total_reward = 0
        episode_penalties = 0
        turn = 0

        while not game_over:
            state = q_agent.get_state(board)
            action = q_agent.choose_action(board, train=True)
            if action is None:
                game_over = True
                reward = -10
                total_reward += reward
                episode_penalties += 1
                break

            row = get_next_open_row(board, action)
            drop_piece(board, row, action, 1 if turn == 0 else 2)

            reward = score_position_q_learning(board, 1 if turn == 0 else 2)
            if winning_move(board, 1 if turn == 0 else 2):
                reward += 50
                game_over = True
            elif is_draw(board):
                reward -= 10
                game_over = True
            else:
                if reward < 0:
                    episode_penalties += 1

            next_state = q_agent.get_state(board)
            q_agent.learn(state, action, reward, next_state)
            total_reward += reward
            turn = 1 - turn

        q_agent.update_epsilon()
        rewards.append(total_reward)
        penalties.append(episode_penalties)
        reward_window.append(total_reward)
        penalty_window.append(episode_penalties)

        if episode % 100 == 0:
            avg_reward = np.mean(reward_window)
            avg_penalties = np.mean(penalty_window)
            print(
                f"Ep: {episode:>5} | Total Reward: {total_reward:>8.2f} | ε: {q_agent.epsilon:>8.4f} "
                f"| Avg Reward (last 50): {avg_reward:>8.2f} | Avg Penalties (last 50): {avg_penalties:>6.2f}"
            )

    print("----- Q-Learning Self-Play Training Complete -----\n")
    plot_rewards(rewards, context="self_play")

    q_agent.save_q_table_qvq("training/q_table(qvq).npy")
    return q_agent


if __name__ == "__main__":
    trained_q_agent = train_q_learning_self_play()
