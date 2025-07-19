import random
import numpy as np
from game.game import (
    get_valid_locations,
    get_next_open_row,
    drop_piece,
    winning_move,
    is_draw,
)
from config import MINIMAX_DEPTH
from agents.score import score_position_q_learning


class MinimaxAgent:
    def __init__(self, depth=MINIMAX_DEPTH):
        # Initialize Minimax agent with a specific search depth
        self.depth = depth

    def minimax(
        self, board, depth, alpha, beta, maximizingPlayer, piece, track_nodes=False
    ):
        # Initialize counters for tracking explored and pruned nodes
        explored_nodes = 0
        pruned_nodes = 0

        # Recursive helper function for Minimax algorithm with Alpha-Beta pruning
        def _minimax(board, depth, alpha, beta, maximizingPlayer, piece):
            nonlocal explored_nodes, pruned_nodes
            explored_nodes += 1  # Count the node as explored

            valid_locations = get_valid_locations(board)  # Get all valid columns
            is_terminal = (
                winning_move(board, 1) or winning_move(board, 2) or is_draw(board)
            )  # Check if the game has reached a terminal state

            # Base case: terminal state or maximum search depth reached
            if depth == 0 or is_terminal:
                if is_terminal:
                    if winning_move(board, piece):  # Win for current player
                        return (None, 1000000)
                    elif winning_move(board, 1 if piece == 2 else 2):  # Opponent win
                        return (None, -1000000)
                    else:  # Draw
                        return (None, 0)
                else:
                    # Return heuristic score if non-terminal state at max depth
                    return (None, score_position_q_learning(board, piece))

            # Maximizing player logic
            if maximizingPlayer:
                value = -np.inf
                best_column = random.choice(valid_locations)
                for col in valid_locations:
                    row = get_next_open_row(board, col)
                    temp_board = board.copy()
                    drop_piece(temp_board, row, col, piece)
                    # Recursively call _minimax for minimizing player
                    new_score = _minimax(
                        temp_board, depth - 1, alpha, beta, False, piece
                    )[1]
                    if new_score > value:
                        value = new_score
                        best_column = col
                    alpha = max(alpha, value)  # Update alpha
                    if alpha >= beta:  # Alpha-Beta pruning
                        pruned_nodes += 1
                        break
                return best_column, value
            else:  # Minimizing player logic
                value = np.inf
                best_column = random.choice(valid_locations)
                opp_piece = 1 if piece == 2 else 2
                for col in valid_locations:
                    row = get_next_open_row(board, col)
                    temp_board = board.copy()
                    drop_piece(temp_board, row, col, opp_piece)
                    # Recursively call _minimax for maximizing player
                    new_score = _minimax(
                        temp_board, depth - 1, alpha, beta, True, piece
                    )[1]
                    if new_score < value:
                        value = new_score
                        best_column = col
                    beta = min(beta, value)  # Update beta
                    if alpha >= beta:  # Alpha-Beta pruning
                        pruned_nodes += 1
                        break
                return best_column, value

        # Execute the minimax algorithm and capture both column and value
        best_column, _ = _minimax(board, depth, alpha, beta, maximizingPlayer, piece)

        # Return result with or without node tracking information
        if track_nodes:
            return best_column, explored_nodes, pruned_nodes
        else:
            return best_column

    def get_move(self, board, piece, track_nodes=False):
        # Entry point for getting the Minimax agent's move
        return self.minimax(
            board, self.depth, -np.inf, np.inf, True, piece, track_nodes=track_nodes
        )
