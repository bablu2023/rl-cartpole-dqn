"""Environment: Gymnasium CartPole wrapped to look like old OpenAI Gym."""
import gymnasium as gym


class OldGymAPI(gym.Wrapper):
    """keras-rl2 was written for the old gym API; Gymnasium uses a newer one.

    Old gym:  reset() -> state            step() -> (state, reward, done, info)
    Gymnasium: reset() -> (state, info)   step() -> (state, reward, terminated, truncated, info)
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # terminated = pole fell / cart left the track; truncated = time limit reached
        return obs, reward, terminated or truncated, info

    def render(self, *args, **kwargs):
        # keras-rl2 calls render(mode=...); Gymnasium's render() takes no arguments
        return self.env.render()


def make_env(env_id="CartPole-v1", render_mode=None):
    """Create a wrapped environment. Use render_mode="rgb_array" to record video."""
    return OldGymAPI(gym.make(env_id, render_mode=render_mode))


def env_sizes(env):
    """Return (number of state values, number of actions)."""
    return env.observation_space.shape[0], env.action_space.n
