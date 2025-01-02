from typing import Optional, Union, Dict

import numpy as np
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID


class MinigridCallback(DefaultCallbacks):
    def __init__(self, path):
        assert path is not None
        super().__init__()
        self.states = None
        self.height = 0
        self.width = 0
        self.path = path

    def on_episode_start(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union["Episode", "EpisodeV2"],
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        self.visited_states = set()
        self.height = base_env.get_sub_environments()[env_index].unwrapped.height
        self.width = base_env.get_sub_environments()[env_index].unwrapped.width
        self.states = np.full((self.width, self.height), 0)

    def on_episode_step(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            episode: "EpisodeV2",
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        env = base_env.get_sub_environments()[env_index].unwrapped
        x, y = env.agent_pos
        self.states[x][y] += 1
        episode.custom_metrics["intrinsic_reward"] = env.intrinsic_reward
        episode.custom_metrics["step_done"] = env.done

    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: EpisodeV2,
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        env = base_env.get_sub_environments()[env_index].unwrapped
        episode.custom_metrics["left"] = env.action_count[0]
        episode.custom_metrics["right"] = env.action_count[1]
        episode.custom_metrics["forward"] = env.action_count[2]
        episode.custom_metrics["pickup"] = env.action_count[3]
        episode.custom_metrics["drop"] = env.action_count[4]
        episode.custom_metrics["toggle"] = env.action_count[5]
        episode.custom_metrics["done"] = env.action_count[6]
        episode.custom_metrics["success_rate"] = env.success_rate
        total_size = self.width * self.height
        # Calculate the number of unique states visited by the agent
        unique_states_visited = np.count_nonzero(self.states)

        # Calculate the percentage of the environment the agent has visited
        percentage_visited = (unique_states_visited / total_size) * 100

        # Log the percentage
        episode.custom_metrics["percentage_visited"] = percentage_visited

        # print(
        #     f"Reward:{episode.total_reward} | env.success_rate:{env.success_rate} | Len:{len(env.success_history)} | env.minNumRooms:{env.minNumRooms}")

    # def on_learn_on_batch(
    #        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    # ) -> None:
    #    seq_lens = train_batch.get("seq_lens", 16)
    #    train_batch['seq_lens'] = np.full_like(seq_lens, 16)
    #    super().on_learn_on_batch(policy=policy, train_batch=train_batch, result=result, **kwargs)
