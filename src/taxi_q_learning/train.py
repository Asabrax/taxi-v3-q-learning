import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from taxi_q_learning.agent import QLearningAgent, QLearningConfig
    from taxi_q_learning.utils import (
        ensure_directory,
        save_reward_plot,
        save_training_dashboard,
        save_training_metrics,
        save_training_summary,
    )
else:
    from .agent import QLearningAgent, QLearningConfig
    from .utils import (
        ensure_directory,
        save_reward_plot,
        save_training_dashboard,
        save_training_metrics,
        save_training_summary,
    )


@dataclass
class TrainingResult:
    rewards: list[float]
    episode_lengths: list[int]
    epsilons: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Q-learning agent on Taxi-v3.")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    return parser.parse_args()


def train(args: argparse.Namespace) -> TrainingResult:
    env = gym.make("Taxi-v3")
    env.action_space.seed(args.seed)

    config = QLearningConfig(
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
    )
    agent = QLearningAgent(
        state_count=env.observation_space.n,
        action_count=env.action_space.n,
        config=config,
        seed=args.seed,
    )

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    epsilons: list[float] = []

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        steps_taken = 0

        for step in range(args.max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.update(state, action, reward, next_state, terminated)

            state = next_state
            total_reward += reward
            steps_taken = step + 1

            if terminated or truncated:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps_taken)
        epsilons.append(agent.config.epsilon)

        if (episode + 1) % 1_000 == 0:
            recent_average = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            print(
                f"Episode {episode + 1:>5} | "
                f"average reward {recent_average:>7.2f} | "
                f"epsilon {agent.config.epsilon:.3f}"
            )

    output_dir = ensure_directory(args.output_dir)
    agent.save(str(output_dir / "q_table.npy"))
    save_reward_plot(episode_rewards, output_dir / "training_rewards.png")
    save_training_dashboard(
        episode_rewards,
        episode_lengths,
        epsilons,
        output_dir / "training_dashboard.png",
    )
    save_training_metrics(
        episode_rewards,
        episode_lengths,
        epsilons,
        output_dir / "training_metrics.csv",
    )
    save_training_summary(
        episode_rewards,
        episode_lengths,
        output_dir / "training_summary.json",
    )
    env.close()

    return TrainingResult(
        rewards=episode_rewards,
        episode_lengths=episode_lengths,
        epsilons=epsilons,
    )


def main() -> None:
    args = parse_args()
    result = train(args)
    rewards = result.rewards
    final_average = sum(rewards[-100:]) / min(100, len(rewards))
    print(f"Training complete. Final 100-episode average reward: {final_average:.2f}")


if __name__ == "__main__":
    main()
