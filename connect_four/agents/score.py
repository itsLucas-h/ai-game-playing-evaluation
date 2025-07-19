from game.game import ROW_COUNT, COLUMN_COUNT


def score_position_q_learning(board, piece):
    # Initialize the score for the board position
    score = 0

    # Evaluate the center column to favor moves closer to the center of the board
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3  # Center-focused moves get a slight boost

    def evaluate_window_q_learning(window, piece):
        # Determine the opponent's piece type
        opp_piece = 1 if piece == 2 else 2
        score = 0

        # Reward highly if we get a "4 in a row"
        if window.count(piece) == 4:
            score += 100  # Immediate winning move boost
        # Moderate reward for "3 in a row" with one open spot
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 5
        # Small reward for "2 in a row" with two open spots
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 2
        # Penalty if the opponent has "3 in a row" with one open spot (defensive play)
        if window.count(opp_piece) == 3 and window.count(0) == 1:
            score -= 6  # Penalty to encourage blocking opponent's potential win
        return score

    # Score rows for potential winning moves
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):  # Only check 4-piece windows
            window = row_array[c : c + 4]
            score += evaluate_window_q_learning(window, piece)

    # Score columns for potential winning moves
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):  # Only check 4-piece windows
            window = col_array[r : r + 4]
            score += evaluate_window_q_learning(window, piece)

    # Score positive diagonal windows
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(4)]
            score += evaluate_window_q_learning(window, piece)

    # Score negative diagonal windows
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r - i][c + i] for i in range(4)]
            score += evaluate_window_q_learning(window, piece)

    return score
