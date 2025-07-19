from agents.random_agent import RandomAgent
from agents.q_learning_agent import QLearningAgent
from agents.score import score_position_q_learning
from game.game import (
    create_board,
    get_next_open_row,
    drop_piece,
    winning_move,
    is_draw,
    is_valid_location,
)
from config import (
    EVALUATION_GAMES,
    Q_LEARNING_ALPHA,
    Q_LEARNING_GAMMA,
    Q_LEARNING_EPSILON,
    Q_LEARNING_EPSILON_DECAY,
    Q_LEARNING_EPSILON_MIN,
)


def evaluate_q_learning_vs_random(q_agent, num_evaluation_games=EVALUATION_GAMES):
    # Initialize Random agent and Q-learning agent with the specified parameters
    random_agent = RandomAgent()
    q_agent = QLearningAgent(
        alpha=Q_LEARNING_ALPHA,
        gamma=Q_LEARNING_GAMMA,
        epsilon=Q_LEARNING_EPSILON,
        epsilon_decay=Q_LEARNING_EPSILON_DECAY,
        epsilon_min=Q_LEARNING_EPSILON_MIN,
    )

    # Initialize metrics to track performance during evaluation
    q_wins, random_wins, draws = 0, 0, 0
    total_moves = 0
    q_total_reward = 0
    q_total_penalties = 0
    q_total_moves = 0

    # Run evaluation games
    for _ in range(num_evaluation_games):
        board = create_board()  # Initialize game board
        game_over = False
        moves = 0
        current_player = 0  # 0 for Q-learning agent, 1 for Random agent

        while not game_over:
            if current_player == 0:  # Q-learning agent's turn
                action = q_agent.choose_action(board, train=False)  # Choose action
                if action is None:  # If no valid action, mark as draw and end game
                    draws += 1
                    game_over = True
                    break

                row = get_next_open_row(board, action)  # Find row for action
                drop_piece(board, row, action, 1)  # Execute action on board
                reward = score_position_q_learning(board, 1)  # Calculate reward

                # Accumulate rewards and penalties for Q-learning agent
                q_total_reward += reward
                q_total_moves += 1
                if reward < 0:
                    q_total_penalties += abs(reward)

                # Check if Q-learning agent wins or if the game is a draw
                if winning_move(board, 1):
                    q_wins += 1
                    game_over = True
                elif is_draw(board):
                    draws += 1
                    game_over = True
            else:  # Random agent's turn
                action = random_agent.get_move(board)  # Random agent chooses action
                if action is not None and is_valid_location(board, action):
                    row = get_next_open_row(board, action)
                    drop_piece(board, row, action, 2)

                    # Check if Random agent wins
                    if winning_move(board, 2):
                        random_wins += 1
                        game_over = True

            # Switch turns between Q-learning and Random agents
            current_player = 1 - current_player
            moves += 1

        total_moves += moves  # Accumulate total moves for all games

    # Calculate metrics for evaluation summary
    q_win_percentage = (q_wins / num_evaluation_games) * 100
    random_win_percentage = (random_wins / num_evaluation_games) * 100
    draw_percentage = (draws / num_evaluation_games) * 100
    avg_reward_per_move = q_total_reward / q_total_moves if q_total_moves else 0
    avg_penalty_per_move = q_total_penalties / num_evaluation_games

    # Print evaluation summary for both agents
    print("\n----- (QvR) Evaluation Summary -----")
    print(f"Q-learning Agent Wins: {q_wins} ({q_win_percentage:.2f}%)")
    print(f"Random Agent Wins: {random_wins} ({random_win_percentage:.2f}%)")
    print(f"Draws: {draws} ({draw_percentage:.2f}%)")
    print(f"Average Reward per Move: {avg_reward_per_move:.2f}")
    print(f"Average Penalties per Move: {avg_penalty_per_move:.2f}\n")
