import faulthandler

import ray
from minigrid.core.actions import Actions
from minigrid.wrappers import RGBImgObsWrapper
from ray.rllib.algorithms import PPOConfig, PPO
from ray.tune import register_env

from trainer import CustomEnv

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, configure_logging=False)
    faulthandler.disable()

    env = RGBImgObsWrapper(CustomEnv(env_type=CustomEnv.Environments.crossing, render_mode="human"))

    # Register the custom environment
    register_env("CustomPlaygroundCrossingEnv-v0",
                 lambda config:
                 RGBImgObsWrapper(CustomEnv(env_type=CustomEnv.Environments.crossing, **config)))

    algo = PPO.from_checkpoint(
        "",
        config=PPOConfig().environment(env="CustomPlaygroundCrossingEnv-v0"))

    for episode in range(0, 10):
        action = 0
        reward = 0
        terminated = False
        obs, _ = env.reset()

        while not terminated:
            action = algo.compute_single_action(observation=obs, prev_action=action, prev_reward=reward)
            obs, reward, terminated, truncated, _ = env.step(action)
            print(f"Action: {Actions(action).name}, Reward: {reward}, Done: {terminated}")
            env.render()
    env.close()
    print("Done!")
