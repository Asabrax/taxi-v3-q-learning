from dataclasses import dataclass

import numpy as np


@dataclass
class QLearningConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.9995
    min_epsilon: float = 0.05


class QLearningAgent:
    def __init__(
        self,
        state_count: int,
        action_count: int,
        config: QLearningConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.state_count = state_count
        self.action_count = action_count
        self.config = config or QLearningConfig()
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((state_count, action_count), dtype=np.float64)

    def choose_action(self, state: int, training: bool = True) -> int:
        should_explore = training and self.rng.random() < self.config.epsilon

        if should_explore:
            return int(self.rng.integers(self.action_count))

        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> None:
        best_next_value = 0.0 if terminated else np.max(self.q_table[next_state])
        old_value = self.q_table[state, action]
        target = reward + self.config.discount_factor * best_next_value

        self.q_table[state, action] = old_value + self.config.learning_rate * (
            target - old_value
        )

    def decay_epsilon(self) -> None:
        self.config.epsilon = max(
            self.config.min_epsilon,
            self.config.epsilon * self.config.epsilon_decay,
        )

    def save(self, path: str) -> None:
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        loaded_table = np.load(path)

        if loaded_table.shape != self.q_table.shape:
            raise ValueError(
                f"Expected Q-table shape {self.q_table.shape}, got {loaded_table.shape}"
            )

        self.q_table = loaded_table
