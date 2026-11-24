"""Train a DQN agent on CartPole and save the weights and a reward plot.

    python train.py                    # defaults: CartPole-v1, 20,000 steps
    python train.py --steps 100000
"""
import argparse
import json
import os
import warnings

import src  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TensorFlow loads)

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # save plots to file, no display needed
import matplotlib.pyplot as plt
import numpy as np

from src.agent import create_agent  # imports keras-rl2 before the model is built
from src.env import env_sizes, make_env


def moving_average(values, window=20):
    if len(values) < window:
        return np.array(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = make_env(args.env)
    states, actions = env_sizes(env)
    print(f"Environment {args.env}: {states} state values, {actions} actions")

    dqn, model = create_agent(states, actions)
    model.summary()

    history = dqn.fit(env, nb_steps=args.steps, visualize=False, verbose=1)
    rewards = [float(r) for r in history.history["episode_reward"]]

    weights_path = os.path.join(args.out, "dqn_weights.h5")
    dqn.save_weights(weights_path, overwrite=True)
    print("Saved weights to", weights_path)

    with open(os.path.join(args.out, "training_rewards.json"), "w") as f:
        json.dump({"env": args.env, "steps": args.steps, "episode_rewards": rewards}, f)

    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.35, label="Episode reward")
    ma = moving_average(rewards)
    plt.plot(range(len(rewards) - len(ma), len(rewards)), ma, label="Moving average (20)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"DQN training on {args.env}")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.out, "training_rewards.png")
    plt.savefig(plot_path, dpi=120)
    print("Saved plot to", plot_path)


if __name__ == "__main__":
    main()
