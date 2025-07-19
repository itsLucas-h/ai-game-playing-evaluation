import os
from game.game import (
    create_board,
    drop_piece,
    get_next_open_row,
    winning_move,
    is_draw,
    get_valid_locations,
    COLUMN_COUNT,
)
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from config import (
    Q_LEARNING_EPSILON_MIN,
    Q_LEARNING_EPSILON_DECAY,
    Q_LEARNING_ALPHA,
    Q_LEARNING_GAMMA,
    Q_LEARNING_EPSILON,
)


def print_board(board):
    print("\n   Connect Four Game Board")
    print("   " + "   ".join(str(i) for i in range(COLUMN_COUNT)))
    print("  +" + "---+" * COLUMN_COUNT)
    for row in board[::-1]:
        formatted_row = " | ".join(
            ["X" if cell == 1 else "O" if cell == 2 else "." for cell in row]
        )
        print(f"  | {formatted_row} |")
        print("  +" + "---+" * COLUMN_COUNT)
    print("\n")


def play_console_game_qvr(load_q_table=False):
    board = create_board()
    q_agent = QLearningAgent(
        alpha=Q_LEARNING_ALPHA,
        gamma=Q_LEARNING_GAMMA,
        epsilon=Q_LEARNING_EPSILON,
        epsilon_decay=Q_LEARNING_EPSILON_DECAY,
        epsilon_min=Q_LEARNING_EPSILON_MIN,
    )

    if load_q_table and os.path.exists("training/q_table(qvr).npy"):
        q_agent.load_q_table_qvr("training/q_table(qvr).npy")
        print("Loaded saved Q-table for Q-learning agent.")
    else:
        print("Starting with a new Q-table for Q-learning agent.")

    random_agent = RandomAgent()
    current_player = 0
    game_over = False

    print("\nStarting Connect Four Game!")
    print("Initial Board:")
    print_board(board)

    while not game_over:
        if current_player == 0:
            # Q-learning agent's turn (Player 1)
            print("\nQ-learning Agent's Turn (Player 1 - X)")
            state = q_agent.get_state(board)
            action = q_agent.choose_action(board, train=False)
            if action is None or action not in get_valid_locations(board):
                print("No valid moves left for Q-learning Agent.")
                game_over = True
                break
            row = get_next_open_row(board, action)
            drop_piece(board, row, action, 1)
            print(f"Q-learning Agent places at column {action}")
            print_board(board)
            if winning_move(board, 1):
                print("\nQ-learning Agent wins!")
                game_over = True
            elif is_draw(board):
                print("\nThe game is a draw!")
                game_over = True
        else:
            # Random agent's turn (Player 2)
            print("\nRandom Agent's Turn (Player 2 - O)")
            action = random_agent.get_move(board)
            if action is None or action not in get_valid_locations(board):
                print("No valid moves left for Random Agent.")
                game_over = True
                break
            row = get_next_open_row(board, action)
            drop_piece(board, row, action, 2)
            print(f"Random Agent places at column {action}")
            print_board(board)
            if winning_move(board, 2):
                print("\nRandom Agent wins!")
                game_over = True
            elif is_draw(board):
                print("\nThe game is a draw!")
                game_over = True

        # Switch turns
        current_player = 1 - current_player

    print("\nFinal Board State:")
    print_board(board)
