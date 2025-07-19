import argparse
from agents.q_learning_agent import QLearningAgent
from config import (
    Q_LEARNING_ALPHA,
    Q_LEARNING_GAMMA,
    Q_LEARNING_EPSILON,
    Q_LEARNING_EPSILON_DECAY,
    Q_LEARNING_EPSILON_MIN,
)
from training.train_qvm import train_q_learning_vs_minimax
from training.train_qvr import train_q_learning_vs_random
from training.train_qvq import train_q_learning_self_play
from evaluate.eval_qvm import evaluate_q_learning_vs_minimax
from evaluate.eval_qvr import evaluate_q_learning_vs_random
from game.play_qvm import play_console_game_qvm
from game.play_qvr import play_console_game_qvr


def main():
    # Set up argument parser for training, evaluation, or game mode selection
    parser = argparse.ArgumentParser(
        description="Train, evaluate, or play a Q-learning agent."
    )
    parser.add_argument(
        "--train",
        choices=["qvm", "qvr", "qvq"],
        help="Specify the agent to train the Q-learning agent against (qvm for minimax, qvr for random, or qvq for self-play).",
    )
    parser.add_argument(
        "--evaluate",
        choices=["qvm", "qvr"],
        help="Specify the agent to evaluate the Q-learning agent against (qvm for minimax, qvr for random).",
    )
    parser.add_argument(
        "--game",
        choices=["play_qvm", "play_qvr"],
        help="Play Q-learning agent against Minimax or Random agent.",
    )
    args = parser.parse_args()

    # Initialize Q-Learning agent with specified configuration parameters
    q_agent = QLearningAgent(
        alpha=Q_LEARNING_ALPHA,
        gamma=Q_LEARNING_GAMMA,
        epsilon=Q_LEARNING_EPSILON,
        epsilon_decay=Q_LEARNING_EPSILON_DECAY,
        epsilon_min=Q_LEARNING_EPSILON_MIN,
    )

    # Training phase: Based on argument, train Q-learning agent against specified opponent
    if args.train == "qvm":
        print("\nTraining Q-learning agent against Minimax agent...")
        trained_q_agent = train_q_learning_vs_minimax()
        trained_q_agent.save_q_table_qvm("training/q_table(qvm).npy")
    elif args.train == "qvr":
        print("\nTraining Q-learning agent against Random agent...")
        trained_q_agent = train_q_learning_vs_random()
        trained_q_agent.save_q_table_qvr("training/q_table(qvr).npy")
    elif args.train == "qvq":
        print("\nTraining Q-learning agent in self-play...")
        trained_q_agent = train_q_learning_self_play()
        trained_q_agent.save_q_table_qvq("training/q_table(qvq).npy")

    # Evaluation phase: Based on argument, evaluate Q-learning agent against specified opponent
    if args.evaluate == "qvm":
        print("\nEvaluating Q-learning agent against Minimax agent...")
        q_agent.load_q_table_qvm("training/q_table(qvm).npy")
        # Uncomment next line to evaluate against self-play Q-table
        # q_agent.load_q_table_qvm("training/q_table(qvq).npy")
        evaluate_q_learning_vs_minimax(q_agent)
    elif args.evaluate == "qvr":
        print("\nEvaluating Q-learning agent against Random agent...")
        q_agent.load_q_table_qvr("training/q_table(qvr).npy")
        # Uncomment next line to evaluate against self-play Q-table
        # q_agent.load_q_table_qvr("training/q_table(qvq).npy")
        evaluate_q_learning_vs_random(q_agent)

    # Game phase: Based on argument, play game between Q-learning agent and specified opponent
    if args.game == "play_qvm":
        print("\nPlaying a game between the Q-learning agent and Minimax agent...")
        q_agent.load_q_table_qvm("training/q_table(qvm).npy")
        play_console_game_qvm(q_agent)

    if args.game == "play_qvr":
        print("\nPlaying a game between the Q-learning agent and Random agent...")
        q_agent.load_q_table_qvr("training/q_table(qvr).npy")
        play_console_game_qvr(q_agent)


# Execute main function if script is run directly
if __name__ == "__main__":
    main()
