from typing import Optional, Union, Dict

from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID


class PacmanCallback(DefaultCallbacks):
    def __init__(self, path):
        assert path is not None
        super().__init__()
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
        pass

    def on_episode_step(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            episode: "EpisodeV2",
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        pass

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
        pass
