"""Load trained weights, test the agent, and record a video.

    python evaluate.py                     # 20 test episodes + results/agent.mp4
    python evaluate.py --episodes 100 --no-video
"""
import argparse
import os
import warnings

import src  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TensorFlow loads)

warnings.filterwarnings("ignore")

import numpy as np

from src.agent import create_agent
from src.env import env_sizes, make_env


def record_video(dqn, env_id, path, episodes=3, fps=30):
    import imageio

    env = make_env(env_id, render_mode="rgb_array")
    frames = []
    for _ in range(episodes):
        state, done = env.reset(), False
        while not done:
            frames.append(env.render())
            q = dqn.compute_q_values([state])  # Q-value for each action
            state, _, done, _ = env.step(int(np.argmax(q)))  # take the best action
    imageio.mimsave(path, frames, fps=fps)
    print(f"Saved video ({len(frames)} frames) to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--weights", default="results/dqn_weights.h5")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--video", default="results/agent.mp4")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    env = make_env(args.env)
    states, actions = env_sizes(env)
    dqn, _ = create_agent(states, actions)
    dqn.load_weights(args.weights)

    scores = dqn.test(env, nb_episodes=args.episodes, visualize=False)
    rewards = scores.history["episode_reward"]
    print(f"Mean reward over {args.episodes} episodes: {np.mean(rewards):.1f} "
          f"(min {np.min(rewards):.0f}, max {np.max(rewards):.0f})")

    if not args.no_video:
        os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
        record_video(dqn, args.env, args.video)


if __name__ == "__main__":
    main()
