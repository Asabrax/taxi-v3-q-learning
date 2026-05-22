import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from taxi_q_learning.evaluate import evaluate
    from taxi_q_learning.train import train
else:
    from .evaluate import evaluate
    from .train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Taxi-v3 Q-learning settings.")
    parser.add_argument("--episodes", type=int, default=8_000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts/experiments")
    return parser.parse_args()


def build_training_args(
    base_args: argparse.Namespace,
    experiment_name: str,
    learning_rate: float,
    discount_factor: float,
    epsilon_decay: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        episodes=base_args.episodes,
        max_steps=base_args.max_steps,
        seed=base_args.seed,
        output_dir=str(Path(base_args.output_dir) / experiment_name),
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        min_epsilon=0.05,
    )


def build_evaluation_args(
    base_args: argparse.Namespace,
    q_table_path: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        episodes=base_args.evaluation_episodes,
        max_steps=base_args.max_steps,
        seed=base_args.seed + 10_000,
        q_table=str(q_table_path),
        render=False,
    )


def main() -> None:
    args = parse_args()
    experiments = [
        ("baseline", 0.10, 0.95, 0.9995),
        ("faster_learning", 0.30, 0.95, 0.9995),
        ("slower_exploration_decay", 0.10, 0.95, 0.9998),
        ("lower_future_discount", 0.10, 0.80, 0.9995),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "experiment_results.csv"

    with results_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "experiment",
                "learning_rate",
                "discount_factor",
                "epsilon_decay",
                "final_training_average_reward",
                "evaluation_average_reward",
                "evaluation_success_rate",
            ]
        )

        for name, learning_rate, discount_factor, epsilon_decay in experiments:
            print(f"Running experiment: {name}")
            training_args = build_training_args(
                args,
                name,
                learning_rate,
                discount_factor,
                epsilon_decay,
            )
            training_result = train(training_args)

            evaluation_args = build_evaluation_args(
                args,
                Path(training_args.output_dir) / "q_table.npy",
            )
            evaluation_rewards = evaluate(evaluation_args)
            evaluation_average = sum(evaluation_rewards) / len(evaluation_rewards)
            success_rate = sum(reward > 0 for reward in evaluation_rewards) / len(
                evaluation_rewards
            )
            final_training_average = sum(training_result.rewards[-100:]) / min(
                100,
                len(training_result.rewards),
            )

            writer.writerow(
                [
                    name,
                    learning_rate,
                    discount_factor,
                    epsilon_decay,
                    round(final_training_average, 3),
                    round(evaluation_average, 3),
                    round(success_rate, 3),
                ]
            )

    print(f"Saved experiment comparison to {results_path}")


if __name__ == "__main__":
    main()
