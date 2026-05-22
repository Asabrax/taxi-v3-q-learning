# Taxi-v3 Q-Learning

In class we worked with the `FrozenLake` environment to learn the basics of reinforcement learning. I wanted to try something a bit more challenging afterwards, so I built this project with Gymnasium's `Taxi-v3` environment.

The idea is simple: a taxi has to pick up a passenger and drop them off at the correct destination. The agent learns this with tabular Q-learning.

I chose Q-learning because `Taxi-v3` still has a discrete state and action space, so it is possible to understand what is happening without using a neural network.

## What The Agent Does

The taxi can:

- move north, south, east, or west
- pick up the passenger
- drop off the passenger

The environment gives rewards and penalties:

- correct drop-off: positive reward
- each step: small negative reward
- illegal pickup/drop-off: bigger negative reward

At first the agent behaves almost randomly. Over many episodes it updates a Q-table and slowly learns which actions are better in each state.

## Setup

```bash
python -m venv myProjk
source myProjk/bin/activate
pip install -r requirements.txt
pip install -e .
```

I pinned Gymnasium below `1.3.0` because newer versions prefer `Taxi-v4`. This project is intentionally using `Taxi-v3`.

## Files

```text
src/taxi_q_learning/
├── agent.py        # Q-table, action selection, Bellman update
├── train.py        # training loop
├── evaluate.py     # evaluates a saved Q-table
├── experiments.py  # small hyperparameter comparison
└── utils.py        # plots and metric saving
```

There are also a few tests in `tests/`.

## Train

```bash
python -m taxi_q_learning.train
```

Training saves files in `artifacts/`:

```text
q_table.npy
training_rewards.png
training_dashboard.png
training_metrics.csv
training_summary.json
```

`q_table.npy` is the learned policy information. The other files are for checking how training went.

## Evaluate

```bash
python -m taxi_q_learning.evaluate
```

To watch one episode in the terminal:

```bash
python -m taxi_q_learning.evaluate --episodes 1 --render
```

My trained agent reached:

```text
Evaluation episodes: 100
Average reward: 7.49
Success rate: 100.0%
```

## Experiments

I compared a few settings for learning rate, discount factor, and epsilon decay:

```bash
python -m taxi_q_learning.experiments
```

Results from one run:

| Experiment | Learning rate | Discount factor | Epsilon decay | Final train avg reward | Eval avg reward | Success rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.10 | 0.95 | 0.9995 | 4.47 | 8.05 | 100% |
| faster_learning | 0.30 | 0.95 | 0.9995 | 4.36 | 8.05 | 100% |
| slower_exploration_decay | 0.10 | 0.95 | 0.9998 | -6.21 | 8.05 | 100% |
| lower_future_discount | 0.10 | 0.80 | 0.9995 | 4.59 | 6.01 | 99% |

The slower epsilon decay keeps the agent exploring for longer, so the training reward looks worse. The lower discount factor still solves most episodes, but the average reward is lower, probably because the routes are less efficient.

## Q-Learning

The Q-table stores one value for each state-action pair:

```text
Q[state, action]
```

The update rule is:

```text
Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s_next)) - Q(s, a))
```

The agent uses epsilon-greedy action selection:

- sometimes choose a random action to explore
- otherwise choose the action with the highest Q-value

Epsilon starts high and decays over time.

## Tests

```bash
pytest
```

The tests cover the Q-table shape, greedy action selection, Q-value updates, and terminal-state updates.

## Notes

This was mainly a project to understand the RL loop better after starting with FrozenLake. Taxi-v3 was useful because it is still small enough for tabular Q-learning, but the task feels more concrete than just moving across a frozen grid.
