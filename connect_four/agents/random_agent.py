from game.game import get_valid_locations
import random
import numpy as np

random.seed(42)
np.random.seed(42)


class RandomAgent:
    def get_move(self, board):
        valid_moves = get_valid_locations(board)
        return random.choice(valid_moves) if valid_moves else None
