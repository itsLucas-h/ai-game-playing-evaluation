# Q-Learning vs Planning in Game Playing – Connect Four AI

## Description

This project implements a Q-learning agent for playing a Connect Four-style game. The agent can be trained via self-play or against predefined opponents like a Minimax agent or a Random agent. You can also evaluate the trained agent or play interactively.

## Requirements

- Python 3.7 or later

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
agents/       # Q-learning, Minimax, and Random agent implementations
game/         # Core game logic and utilities
training/     # Scripts for training the Q-learning agent
evaluate/     # Scripts for evaluating agent performance
main.py       # Main entry point to run training, evaluation, or gameplay
config.py     # Configuration and hyperparameter settings
```

## Usage

Run the `main.py` script with one of the following options:

### 1. Train the Q-learning Agent

- Train against Minimax:
  ```bash
  python main.py --train qvm
  ```

- Train against Random:
  ```bash
  python main.py --train qvr
  ```

- Train using self-play:
  ```bash
  python main.py --train qvq
  ```

### 2. Evaluate the Q-learning Agent

- Evaluate against Minimax:
  ```bash
  python main.py --evaluate qvm
  ```

- Evaluate against Random:
  ```bash
  python main.py --evaluate qvr
  ```

### 3. Train and Evaluate in Sequence

- Train against Random and evaluate against Random:
  ```bash
  python main.py --train qvr --evaluate qvr
  ```

- Train against Minimax and evaluate against Minimax:
  ```bash
  python main.py --train qvm --evaluate qvm
  ```

### 4. Play a Game

- Q-learning vs Minimax:
  ```bash
  python main.py --game play_qvm
  ```

- Q-learning vs Random:
  ```bash
  python main.py --game play_qvr
  ```

## Q-table Saving and Loading

Q-tables are saved automatically during training and loaded during evaluation or gameplay:

- `q_table(qvm).npy` – Trained vs Minimax
- `q_table(qvr).npy` – Trained vs Random
- `q_table(qvq).npy` – Self-play training

The correct Q-table is automatically loaded based on the mode selected.

## Features

- Q-learning agent with configurable hyperparameters  
- Comparison against baseline Minimax and Random agents  
- Automatic Q-table persistence  
- Command-line control for training, evaluation, and gameplay
