# Taxi-v3 Q-Learning Agent

A compact reinforcement learning project that trains a tabular Q-learning agent to solve the classic `Taxi-v3` environment from Gymnasium.

Note: newer Gymnasium releases promote `Taxi-v4`, so this project pins Gymnasium to a compatible version range for `Taxi-v3`.

This project is designed to be easy to explain on a CV or in an interview: the algorithm is implemented from scratch, the training loop is reproducible, and the outputs show learning progress clearly.

## Project Highlights

- Implements tabular Q-learning with epsilon-greedy exploration.
- Trains and evaluates on Gymnasium `Taxi-v3`.
- Saves the learned Q-table and reward plots.
- Exports training metrics as CSV and JSON for analysis.
- Compares multiple hyperparameter settings.
- Includes a readable code structure and line-by-line explanation.
- Adds lightweight tests for the agent logic.

## What Taxi-v3 Is

`Taxi-v3` is a grid-world reinforcement learning task. The taxi must pick up a passenger from one location and drop them off at the correct destination.

The agent receives:

- Positive reward for a successful drop-off.
- Negative reward for each time step.
- Larger penalties for illegal pickup or drop-off actions.

The goal is to learn a policy that completes trips quickly while avoiding illegal actions.

## Folder Structure

```text
myProjk/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── line_by_line_explanation.md
├── src/
│   └── taxi_q_learning/
│       ├── __init__.py
│       ├── agent.py
│       ├── experiments.py
│       ├── evaluate.py
│       ├── train.py
│       └── utils.py
└── tests/
    └── test_agent.py
```

## Setup

From the project folder:

```bash
source myProjk/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you upload only this project folder to GitHub, create the environment inside the project instead:

```bash
python -m venv myProjk
source myProjk/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Train

```bash
python -m taxi_q_learning.train
```

This creates:

```text
artifacts/q_table.npy
artifacts/training_rewards.png
artifacts/training_dashboard.png
artifacts/training_metrics.csv
artifacts/training_summary.json
```

## Run Experiments

```bash
python -m taxi_q_learning.experiments
```

This trains several Q-learning configurations and writes:

```text
artifacts/experiments/experiment_results.csv
```

The comparison includes learning rate, discount factor, epsilon decay, final training reward, evaluation reward, and success rate.

## Evaluate

```bash
python -m taxi_q_learning.evaluate
```

To watch one rendered episode in the terminal:

```bash
python -m taxi_q_learning.evaluate --render
```

## Run Tests

```bash
pytest
```

## Results

Using the default configuration for 20,000 training episodes, the trained policy reached:

```text
Evaluation episodes: 100
Average reward: 7.49
Success rate: 100.0%
```

The reward curve starts strongly negative because the agent explores random moves, illegal pickups, and inefficient routes. As epsilon decays, the agent exploits the learned Q-table more often, reaches the passenger and destination faster, and stabilizes at a positive reward.

## Experiment Design

The project includes a small hyperparameter study:

- `baseline`: balanced learning rate and exploration decay.
- `faster_learning`: higher learning rate to adapt more aggressively.
- `slower_exploration_decay`: keeps exploration high for longer.
- `lower_future_discount`: values future rewards less strongly.

This makes the project easier to discuss as an applied AI experiment: not only “I trained a model,” but “I compared training settings and evaluated their effect.”

Default experiment results over 8,000 training episodes and 100 evaluation episodes:

| Experiment | Learning rate | Discount factor | Epsilon decay | Final train avg reward | Eval avg reward | Success rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.10 | 0.95 | 0.9995 | 4.47 | 8.05 | 100% |
| faster_learning | 0.30 | 0.95 | 0.9995 | 4.36 | 8.05 | 100% |
| slower_exploration_decay | 0.10 | 0.95 | 0.9998 | -6.21 | 8.05 | 100% |
| lower_future_discount | 0.10 | 0.80 | 0.9995 | 4.59 | 6.01 | 99% |

The slower exploration schedule performs worse during training because the agent keeps taking random actions for longer. The lower discount factor reaches a high success rate, but its lower average reward suggests less efficient routes.

## CV Description

Built and evaluated a reinforcement learning agent for Gymnasium Taxi-v3 using tabular Q-learning. Implemented epsilon-greedy exploration, Bellman Q-value updates, model checkpointing, metric logging, reward and episode-length visualization, hyperparameter comparison, and unit tests in a reproducible Python project.

## Algorithm Summary

Q-learning learns an action-value table `Q[state, action]`.

At each step, the agent:

1. Chooses an action using epsilon-greedy exploration.
2. Receives a reward and observes the next state.
3. Updates the Q-value using the Bellman equation:

```text
Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s_next)) - Q(s, a))
```

After enough training, the best action for a state is the action with the largest Q-value.
