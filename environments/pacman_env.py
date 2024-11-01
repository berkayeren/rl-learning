try:
    import gymnasium as gym

    gymnasium = True
except Exception:
    import gym

    gymnasium = False


class PacmanWrapper(gym.Wrapper):
    def __init__(self, env, prediction_net=None,
                 prediction_criterion=None,
                 prediction_optimizer=None, ):
        super(PacmanWrapper, self).__init__(env)
        self.env = env
        self.custom_reward = 0
        self.prediction_net = prediction_net
        self.prediction_criterion = prediction_criterion
        self.prediction_optimizer = prediction_optimizer
        self.episode_history = []

    def step(self, action):
        obs, reward, is_terminal, is_truncated, info = self.env.step(action)

        return obs, reward, is_terminal, is_truncated, info

    def reset(self, **kwargs):
        return self.env.render()
