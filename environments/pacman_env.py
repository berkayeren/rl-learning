import gymnasium as gym


class PacmanWrapper(gym.Wrapper):
    def __init__(self, env):
        super(PacmanWrapper, self).__init__(env)
