import numpy as np

from classes.policy import Policy
from classes.simulation import Simulation


def train_policy(env, num_episodes=10000, weight=0.1, discount_factor=0.9):
    """Training a policy by updating it with rollout experiences."""
    policy = Policy(env)
    sim = Simulation(env)
    for _ in range(num_episodes):
        experiences = sim.rollout(policy)
        update_policy(policy, experiences, weight, discount_factor)

    return policy


def update_policy(policy, experiences, weight=0.1, discount_factor=0.9):
    """Updates a given policy with a list of (state, action, reward, state)
    experiences."""
    for state, action, reward, next_state in experiences:
        next_max = np.max(policy.state_action_table[next_state])

        value = policy.state_action_table[state][action]

        new_value = (1 - weight) * value + weight * \
                    (reward + discount_factor * next_max)

        policy.state_action_table[state][action] = new_value


def evaluate_policy(env, policy, num_episodes=10):
    """Evaluate a trained policy through rollouts."""
    simulation = Simulation(env)
    steps = 0

    for _ in range(num_episodes):
        experiences = simulation.rollout(policy, render=True, explore=False)

        steps += len(experiences)

    print(f"{steps / num_episodes} steps on average "
          f"for a total of {num_episodes} episodes.")

    return steps / num_episodes


if __name__ == "__main__":
    from classes.environment import Environment

    environment = Environment()
    untrained_policy = Policy(environment)
    trained_policy = train_policy(environment)

    evaluate_policy(environment, trained_policy)
    # evaluate_policy(environment, untrained_policy)
