import numpy as np
import pytest

from taxi_q_learning.agent import QLearningAgent, QLearningConfig


def test_q_table_has_expected_shape() -> None:
    agent = QLearningAgent(state_count=500, action_count=6)

    assert agent.q_table.shape == (500, 6)


def test_greedy_action_uses_highest_q_value() -> None:
    agent = QLearningAgent(state_count=2, action_count=3)
    agent.q_table[1] = np.array([0.0, 5.0, 1.0])

    action = agent.choose_action(state=1, training=False)

    assert action == 1


def test_update_applies_q_learning_rule() -> None:
    config = QLearningConfig(learning_rate=0.5, discount_factor=0.9)
    agent = QLearningAgent(state_count=2, action_count=2, config=config)
    agent.q_table[1] = np.array([2.0, 4.0])

    agent.update(state=0, action=1, reward=10, next_state=1, terminated=False)

    expected = 0.0 + 0.5 * ((10 + 0.9 * 4.0) - 0.0)
    assert agent.q_table[0, 1] == pytest.approx(expected)


def test_terminated_update_ignores_future_value() -> None:
    config = QLearningConfig(learning_rate=1.0, discount_factor=0.9)
    agent = QLearningAgent(state_count=2, action_count=2, config=config)
    agent.q_table[1] = np.array([100.0, 100.0])

    agent.update(state=0, action=0, reward=20, next_state=1, terminated=True)

    assert agent.q_table[0, 0] == pytest.approx(20.0)
