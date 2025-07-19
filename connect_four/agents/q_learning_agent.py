import numpy as np
from game.game import (
    COLUMN_COUNT,
    get_valid_locations,
)
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)


class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
        # Initialize Q-learning agent with parameters
        self.q_table = {}  # Initialize Q-table as a dictionary
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_state(self, board):
        # Convert the game board to a flattened string to use as a Q-table key
        return str(board.flatten())

    def choose_action(self, board, train=False):
        # Choose an action based on epsilon-greedy policy
        valid_moves = get_valid_locations(board)
        if not valid_moves:
            return None  # No valid moves available

        state = self.get_state(board)  # Get state representation
        if (
            train and np.random.rand() < self.epsilon
        ):  # Explore with probability epsilon
            return random.choice(valid_moves)
        else:
            # Exploit by choosing action with the highest Q-value for the state
            q_values = self.q_table.get(state, np.zeros(COLUMN_COUNT))
            return max(valid_moves, key=lambda x: q_values[x])

    def learn(self, state, action, reward, next_state):
        # Update Q-table using Q-learning update rule
        if state not in self.q_table:
            self.q_table[state] = np.zeros(
                COLUMN_COUNT
            )  # Initialize Q-values for state
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(
                COLUMN_COUNT
            )  # Initialize Q-values for next state

        current_q = self.q_table[state][action]  # Current Q-value
        max_future_q = np.max(self.q_table[next_state])  # Max Q-value for next state
        # Update Q-value based on the Q-learning formula
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_future_q - current_q
        )

    def update_epsilon(self):
        # Decay epsilon to gradually reduce exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Save and load Q-table for different training contexts
    def save_q_table_qvr(self, filename="q_table(qvr).npy"):
        np.save(filename, self.q_table)

    def load_q_table_qvr(self, filename="q_table(qvr).npy"):
        self.q_table = np.load(filename, allow_pickle=True).item()

    def save_q_table_qvm(self, filename="q_table(qvm).npy"):
        np.save(filename, self.q_table)

    def load_q_table_qvm(self, filename="q_table(qvm).npy"):
        self.q_table = np.load(filename, allow_pickle=True).item()

    def save_q_table_qvq(self, filename="q_table(qvq).npy"):
        np.save(filename, self.q_table)

    def load_q_table_qvq(self, filename="q_table(qvq).npy"):
        self.q_table = np.load(filename, allow_pickle=True).item()
