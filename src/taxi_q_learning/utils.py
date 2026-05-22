import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def moving_average(values: list[float], window_size: int) -> np.ndarray:
    if len(values) < window_size:
        return np.array(values, dtype=np.float64)

    weights = np.ones(window_size) / window_size
    return np.convolve(values, weights, mode="valid")


def save_reward_plot(rewards: list[float], output_path: str | Path) -> None:
    reward_average = moving_average(rewards, window_size=100)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.25, label="Episode reward")
    plt.plot(
        range(len(reward_average)),
        reward_average,
        color="tab:blue",
        label="100-episode moving average",
    )
    plt.title("Taxi-v3 Q-Learning Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_training_metrics(
    rewards: list[float],
    episode_lengths: list[int],
    epsilons: list[float],
    output_path: str | Path,
) -> None:
    with Path(output_path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "reward", "episode_length", "epsilon"])

        for episode, values in enumerate(zip(rewards, episode_lengths, epsilons), start=1):
            reward, episode_length, epsilon = values
            writer.writerow([episode, reward, episode_length, epsilon])


def save_training_summary(
    rewards: list[float],
    episode_lengths: list[int],
    output_path: str | Path,
) -> None:
    recent_rewards = rewards[-100:]
    recent_lengths = episode_lengths[-100:]
    summary = {
        "episodes": len(rewards),
        "final_100_episode_average_reward": float(np.mean(recent_rewards)),
        "final_100_episode_average_length": float(np.mean(recent_lengths)),
        "best_episode_reward": float(np.max(rewards)),
        "worst_episode_reward": float(np.min(rewards)),
    }

    with Path(output_path).open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)


def save_training_dashboard(
    rewards: list[float],
    episode_lengths: list[int],
    epsilons: list[float],
    output_path: str | Path,
) -> None:
    reward_average = moving_average(rewards, window_size=100)
    length_average = moving_average(episode_lengths, window_size=100)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(rewards, alpha=0.2, label="Episode reward")
    axes[0].plot(reward_average, color="tab:blue", label="100-episode average")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].plot(episode_lengths, alpha=0.2, color="tab:orange", label="Episode length")
    axes[1].plot(length_average, color="tab:red", label="100-episode average")
    axes[1].set_ylabel("Steps")
    axes[1].legend()

    axes[2].plot(epsilons, color="tab:green", label="Epsilon")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Exploration")
    axes[2].legend()

    fig.suptitle("Taxi-v3 Training Metrics")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
