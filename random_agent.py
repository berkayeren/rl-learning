import gym
from gym.wrappers import TimeLimit


def run_random_episode(env: TimeLimit) -> None:
    """
    Runs a single episode using random actions and prints observations.

    Args:
    - env (TimeLimit): The Gym environment.

    Returns:
    - None
    """
    observation = env.reset()

    while True:
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            break

    env.close()


if __name__ == "__main__":
    env = gym.make("BipedalWalker-v3")
    run_random_episode(env)
