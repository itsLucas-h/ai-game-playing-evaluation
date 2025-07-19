import numpy as np
import matplotlib.pyplot as plt


def plot_rewards(rewards, context="minimax"):
    plt.figure(figsize=(10, 6))
    moving_avg = np.convolve(rewards, np.ones(50) / 50, mode="valid")

    plt.plot(moving_avg, label="Moving Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    if context == "minimax":
        plt.title("Total Reward per Episode: Q-Learning Agent vs Minimax")
    elif context == "random":
        plt.title("Total Reward per Episode: Q-Learning Agent vs Random")
    elif context == "self_play":
        plt.title("Total Reward per Episode: Q-Learning Agent Self-Play")
    else:
        plt.title("Total Reward per Episode: Q-Learning Agent")

    plt.legend()
    plt.grid()
    plt.show()
