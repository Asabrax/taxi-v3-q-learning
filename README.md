# Taxi-v3 Q-Learning

This is a small reinforcement learning project using the `Taxi-v3` environment from Gymnasium.

The goal is to train a taxi agent to pick up a passenger and drop them off at the correct destination. I used tabular Q-learning because the environment has a small discrete state space, so a neural network is not really necessary here.

The project includes training, evaluation, saved metrics, plots, a small hyperparameter comparison, and a few tests for the agent logic.

## Environment

`Taxi-v3` is a grid-world task. The agent can move around the map, pick up a passenger, and drop the passenger off.

Rewards:

- successful drop-off gives a positive reward
- every step has a small negative reward
- illegal pickup or drop-off actions get a larger penalty

The agent starts with no knowledge of the environment and learns better actions by updating a Q-table.

## Setup

From the project folder:

```bash
python -m venv myProjk
source myProjk/bin/activate
pip install -r requirements.txt
pip install -e .
```

I pinned Gymnasium below `1.3.0` because newer versions prefer `Taxi-v4`, while this project is specifically for `Taxi-v3`.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── taxi_q_learning/
│       ├── agent.py
│       ├── train.py
│       ├── evaluate.py
│       ├── experiments.py
│       └── utils.py
├── tests/
│   └── test_agent.py
└── docs/
    └── github_upload.md
```

## Training

Run:

```bash
python -m taxi_q_learning.train
```

This trains the agent and writes the outputs to `artifacts/`:

```text
q_table.npy
training_rewards.png
training_dashboard.png
training_metrics.csv
training_summary.json
```

The Q-table is saved so the agent can be evaluated later without training again.

## Evaluation

Run:

```bash
python -m taxi_q_learning.evaluate
```

To see one episode rendered in the terminal:

```bash
python -m taxi_q_learning.evaluate --episodes 1 --render
```

With the default training setup, I got:

```text
Evaluation episodes: 100
Average reward: 7.49
Success rate: 100.0%
```

## Experiments

I also compared a few hyperparameter settings:

```bash
python -m taxi_q_learning.experiments
```

The results are saved to:

```text
artifacts/experiments/experiment_results.csv
```

Results from my run:

| Experiment | Learning rate | Discount factor | Epsilon decay | Final train avg reward | Eval avg reward | Success rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.10 | 0.95 | 0.9995 | 4.47 | 8.05 | 100% |
| faster_learning | 0.30 | 0.95 | 0.9995 | 4.36 | 8.05 | 100% |
| slower_exploration_decay | 0.10 | 0.95 | 0.9998 | -6.21 | 8.05 | 100% |
| lower_future_discount | 0.10 | 0.80 | 0.9995 | 4.59 | 6.01 | 99% |

The slower exploration decay keeps the agent exploring for longer, so the training reward is worse even though the final evaluated policy still works. The lower discount factor also works, but the average reward is lower, which suggests the learned routes are less efficient.

## Q-Learning Update

The agent stores values in a Q-table:

```text
Q[state, action]
```

After each action, it updates the table with:

```text
Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s_next)) - Q(s, a))
```

Where:

- `alpha` is the learning rate
- `gamma` is the discount factor
- `max(Q(s_next))` is the best estimated future value

During training the agent uses epsilon-greedy action selection. At the beginning it explores a lot, and over time it relies more on the learned Q-values.

## Tests

Run:

```bash
pytest
```

The tests check the Q-table shape, greedy action selection, the Q-learning update, and terminal-state behavior.

## What I Learned

This project helped me understand how tabular reinforcement learning works without hiding the important parts behind a library model. The most useful parts were implementing the Bellman update, seeing how epsilon decay changes learning, and comparing how different hyperparameters affect training and evaluation.
