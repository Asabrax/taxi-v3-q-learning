import argparse
import sys
from pathlib import Path

import gymnasium as gym

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from taxi_q_learning.agent import QLearningAgent
else:
    from .agent import QLearningAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Taxi-v3 agent.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--q-table", type=str, default="artifacts/q_table.npy")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> list[float]:
    render_mode = "ansi" if args.render else None
    env = gym.make("Taxi-v3", render_mode=render_mode)

    agent = QLearningAgent(
        state_count=env.observation_space.n,
        action_count=env.action_space.n,
        seed=args.seed,
    )
    agent.load(args.q_table)

    episode_rewards: list[float] = []

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        total_reward = 0.0

        for step in range(args.max_steps):
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if args.render:
                print(f"Episode {episode + 1}, step {step + 1}")
                print(env.render())

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

    env.close()
    return episode_rewards


def main() -> None:
    args = parse_args()

    if not Path(args.q_table).exists():
        raise FileNotFoundError(
            f"Could not find {args.q_table}. Train first with: "
            "python -m taxi_q_learning.train"
        )

    rewards = evaluate(args)
    average_reward = sum(rewards) / len(rewards)
    success_rate = sum(reward > 0 for reward in rewards) / len(rewards)

    print(f"Evaluation episodes: {len(rewards)}")
    print(f"Average reward: {average_reward:.2f}")
    print(f"Success rate: {success_rate:.1%}")


if __name__ == "__main__":
    main()
