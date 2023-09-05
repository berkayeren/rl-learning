import time


class Simulation(object):
    def __init__(self, env):
        """Simulates rollouts of an environment, given a policy to follow."""
        self.env = env

    def rollout(self, policy, render=False, explore=True, epsilon=0.1):
        """Returns experiences for a policy rollout."""
        experiences = []
        state = self.env.reset()
        done = False
        while not done:
            action = policy.get_action(state, explore, epsilon)
            next_state, reward, done, info = self.env.step(action)
            experiences.append([state, action, reward, next_state])
            state = next_state
            if render:
                time.sleep(0.05)
                self.env.render()

        return experiences
