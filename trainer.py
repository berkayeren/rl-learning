"""
trainer.py — PPO training entry point for MiniGrid exploration experiments.

This module wires together a configurable MiniGrid environment, several intrinsic
motivation methods, and Ray RLlib's PPO algorithm to study how different
exploration bonuses affect an agent's ability to solve sparse-reward grid worlds.

High-level structure
--------------------
* ``CustomCallback`` — an RLlib callback that records per-episode diagnostic
  metrics (action counts, intrinsic/shaped reward, map coverage).
* ``plot_heatmap`` — utility that renders an agent visit-count heatmap with
  walls, doors, and the goal overlaid.
* ``CustomEnv`` — a single MiniGrid environment subclass that can be configured
  (via ``env_type``) to build any of several hand-designed layouts (empty,
  crossing, four-rooms, multi-room, multi-room-with-keys, twelve-rooms, long
  corridor). It also computes the intrinsic reward each step using whichever
  exploration method is enabled (DoWhaM v1/v2, count-based, or RND) and shapes
  the extrinsic reward.
* ``custom_trial_name`` — produces human-readable Ray Tune trial names.
* ``__main__`` — parses CLI arguments, registers the environment under the
  chosen observation wrapper, builds a PPO config, and launches either a grid
  search "experiment" or a "hyperparameter_search".

Coordinate convention
---------------------
MiniGrid uses ``(x, y)`` integer grid coordinates with the origin at the
top-left. The agent observes a 7×7 egocentric window (see ``VIEW_SIZE``) and the
DoWhaM v2 reward needs to map cells of that window back to global coordinates;
``BASE_OFFSETS`` plus ``transform_coords`` handle that egocentric→global mapping.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
from enum import IntEnum
from typing import Union, Optional, Dict

import gymnasium as gym
import numpy as np
import ray
import torch
from gymnasium import spaces
from matplotlib import pyplot as plt
from minigrid.core.constants import COLOR_NAMES

# --- Egocentric-view geometry -------------------------------------------------
# The agent sees a VIEW_SIZE×VIEW_SIZE window. The agent itself sits at the
# bottom-center of that window (it always looks "up" within its own view), so we
# anchor the offset grid there.
VIEW_SIZE = 7
# Pivot at bottom-center of the egocentric view (row 6, col 3)
PIVOT_ROW = VIEW_SIZE - 1
PIVOT_COL = VIEW_SIZE // 2
# BASE_OFFSETS[i, j] gives the (dx, dy) displacement, relative to the agent, of
# the cell at row ``i`` / col ``j`` of the egocentric window, assuming the agent
# faces East (agent_dir == 0). ``dx`` runs forward from the agent (rows closer
# to the top of the view are farther ahead) and ``dy`` runs left→right across
# the view. ``transform_coords`` rotates these base offsets for the other three
# facing directions.
BASE_OFFSETS = np.array([
    [(PIVOT_ROW - i, j - PIVOT_COL) for j in range(VIEW_SIZE)]
    for i in range(VIEW_SIZE)
])
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Lava, Wall, Door, Key, Floor
from minigrid.envs import MultiRoom
from minigrid.wrappers import RGBImgObsWrapper, FullyObsWrapper, FlatObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper
from ray import tune, train
from ray.air import FailureConfig
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms import PPOConfig, ImpalaConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module import RLModule
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID
from ray.tune import register_env, CheckpointConfig
from ray.tune.search import BasicVariantGenerator

from environments.minigrid_wrapper import PositionBasedWrapper
from environments.empty import EmptyEnv
from intrinsic_motivation.count_based import CountExploration
from intrinsic_motivation.dowham_v2 import DoWhaMIntrinsicRewardV2
from intrinsic_motivation.dowham_v1 import DoWhaMIntrinsicRewardV1
from intrinsic_motivation.rnd import RNDModule

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX


class CustomCallback(RLlibCallback):
    """RLlib callback that logs per-episode exploration diagnostics.

    RLlib invokes the ``on_episode_*`` hooks on each env-runner as rollouts are
    collected. This callback writes values onto ``episode.custom_metrics`` so
    they are aggregated and surfaced in the training results (and any connected
    logger such as TensorBoard). The metrics it records are:

    * per-action counters (``left``, ``right``, ``forward``, ``pickup``,
      ``drop``, ``toggle``, ``done``) — how often the policy chose each action;
    * ``intrinsic_reward`` / ``shaped_reward`` / ``termination_reward`` — the
      reward components computed by :class:`CustomEnv` on the latest step;
    * ``step_done`` — whether the environment terminated on this step;
    * ``percentage_visited`` — fraction of the grid the agent touched this
      episode (computed by the env on ``reset``);
    * ``percentage_history`` — number of successful terminations in the env's
      rolling 100-episode window.

    The custom-metric values are pulled from the *unwrapped* sub-environment,
    which is the :class:`CustomEnv` instance carrying the attributes above.
    """

    def on_episode_start(
            self,
            *,
            episode: Union[EpisodeType, EpisodeV2],
            env_runner: Optional["EnvRunner"] = None,
            metrics_logger: Optional[MetricsLogger] = None,
            env: Optional[gym.Env] = None,
            env_index: int,
            rl_module: Optional[RLModule] = None,
            # TODO (sven): Deprecate these args.
            worker: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            **kwargs,
    ) -> None:
        """Initialize the per-action counters at the start of every episode.

        The counters are zeroed here so that ``on_episode_step`` can increment
        the one matching the action actually taken.
        """
        episode.custom_metrics["left"] = 0
        episode.custom_metrics["right"] = 0
        episode.custom_metrics["forward"] = 0
        episode.custom_metrics["pickup"] = 0
        episode.custom_metrics["drop"] = 0
        episode.custom_metrics["toggle"] = 0
        episode.custom_metrics["done"] = 0

    def on_episode_step(
            self,
            *,
            episode: Union[EpisodeType, EpisodeV2],
            env_runner: Optional["EnvRunner"] = None,
            metrics_logger: Optional[MetricsLogger] = None,
            env: Optional[gym.Env] = None,
            env_index: int,
            rl_module: Optional[RLModule] = None,
            # TODO (sven): Deprecate these args.
            worker: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            **kwargs,
    ) -> None:
        """Record reward components and tally the action taken on this step.

        Reads the freshly updated reward fields off the unwrapped env and
        increments the counter named after the action enum (e.g. ``forward``).
        """
        env = base_env.get_sub_environments()[env_index].unwrapped
        episode.custom_metrics["intrinsic_reward"] = env.intrinsic_reward
        episode.custom_metrics["shaped_reward"] = env.shaped_reward
        episode.custom_metrics["step_done"] = env.done
        episode.custom_metrics[env.actions(env.action).name] += 1
        episode.custom_metrics["termination_reward"] = env.termination_reward

    def on_episode_end(
            self,
            *,
            episode: Union[EpisodeType, EpisodeV2],
            env_runner: Optional["EnvRunner"] = None,
            metrics_logger: Optional[MetricsLogger] = None,
            env: Optional[gym.Env] = None,
            env_index: int,
            rl_module: Optional[RLModule] = None,
            # TODO (sven): Deprecate these args.
            worker: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            **kwargs,
    ) -> None:
        """Record episode-level coverage stats once the episode finishes.

        ``percentage_visited`` is the share of the grid the agent touched, and
        ``percentage_history`` is the count of successful terminations in the
        env's rolling 100-episode window — both maintained by the env.
        """
        env = base_env.get_sub_environments()[env_index].unwrapped
        episode.custom_metrics["percentage_visited"] = env.percentage_visited
        episode.custom_metrics["percentage_history"] = env.percentage_history.count(True)


def plot_heatmap(env, filename="visit_heatmap.png"):
    """Render and save a heatmap of how often each grid cell was visited.

    The agent's per-cell visit counts (``env.states``) are drawn as a "hot"
    colormap, with walls (black squares), doors (yellow diamonds), and the goal
    (green star) overlaid for context. The figure is written to the project's
    ``heatmaps/`` directory.

    Args:
        env: A :class:`CustomEnv` (unwrapped) exposing ``states`` (the visit
            grid), ``grid``, ``width``/``height``, and optionally ``goal_pos``.
        filename: Output file name; saved under ``heatmaps/``.
    """
    heatmap_data = np.flipud(env.states.T)  # Flip for correct orientation

    plt.figure(figsize=(6, 6))
    plt.title("Agent Visit Heatmap")

    # Plot heatmap with visit counts
    plt.imshow(heatmap_data, cmap="hot", origin="lower", alpha=0.5)  # Heatmap semi-transparent

    for x in range(env.width):
        for y in range(env.height):
            count = env.states[x, y]
            if count > 0:  # Only show counts for visited cells
                plt.text(x, env.height - y - 1, "", ha='center', va='center', color='white', fontsize=8)

    # Overlay walls in black
    for x in range(env.width):
        for y in range(env.height):
            if isinstance(env.grid.get(x, y), Wall):  # Check if cell is a wall
                plt.scatter(x, env.height - y - 1, color='black', s=40, marker='s')
            if isinstance(env.grid.get(x, y), Door):  # Check if cell is a wall
                plt.scatter(x, env.height - y - 1, color="yellow", s=80, marker='D', edgecolors="black",
                            linewidth=1.2)

    # # Overlay explicit key icons at given locations
    # keys_to_plot = [
    #     (2, 3, "red"),
    #     (5, 7, "green"),
    #     (9, 11, "blue"),
    # ]
    # for kx, ky, kc in keys_to_plot:
    #     if 0 <= kx < env.width and 0 <= ky < env.height:
    #         plt.scatter(kx, env.height - ky - 1, s=120, marker="P",
    #                     color=kc, edgecolors="black", linewidth=1.0, zorder=5)
    #         # Optional: add a small 'K' label
    #         plt.text(kx, env.height - ky - 1, "K", ha="center", va="center",
    #                  color="white", fontsize=9, fontweight="bold", zorder=6)

    # Overlay goal position in green
    if hasattr(env, "goal_pos") and env.goal_pos:
        goal_x, goal_y = env.goal_pos
        plt.scatter(goal_x, env.height - goal_y - 1, color='lime', s=100, marker='*', edgecolors="black", linewidth=1.5)

    # Add colorbar
    plt.colorbar(label="Visit Count")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)

    # Save the image
    plt.savefig("/Users/berkayeren/PycharmProjects/rl-learning/heatmaps/" + filename)
    plt.close()


class CustomEnv(EmptyEnv):
    """Configurable MiniGrid environment with pluggable intrinsic motivation.

    A single environment class that can build any of several hand-designed grid
    layouts and augment the sparse extrinsic reward with one of several
    exploration bonuses. Which layout is built is selected by ``env_type`` (see
    the :class:`Environments` enum); which intrinsic reward is active is selected
    by the ``enable_*`` flags (at most one is normally enabled).

    Per step it:
      1. records the visited cell into ``self.states`` (for the heatmap/coverage);
      2. computes an intrinsic reward via the enabled method;
      3. shapes the extrinsic reward with a success bonus and the (scaled)
         intrinsic reward.

    Key instance attributes (read by :class:`CustomCallback`):
        intrinsic_reward: Latest intrinsic bonus from the active method.
        shaped_reward: Total shaping added to the env reward this step.
        termination_reward: Success bonus added on the terminating step.
        done: Whether the env terminated on the latest step.
        percentage_visited: Grid coverage (%) from the previous episode.
        percentage_history: Rolling deque (maxlen 100) of episode success flags.
        states: ``(width, height)`` int array of per-cell visit counts.
    """

    class Environments(IntEnum):
        """Enumeration of the available grid layouts.

        The integer values are the codes passed through ``env_config`` and CLI
        ``--environment``; each maps to a ``_gen_grid`` builder method.
        """
        empty = 0
        crossing = 1
        four_rooms = 2
        multi_room = 3
        multi_room_key = 5
        twelve_rooms = 4
        long_corridor = 6

    def __init__(self, **kwargs):
        """Pull configuration out of ``kwargs`` and initialize exploration state.

        Recognized ``kwargs`` (all optional, with defaults) include:
          * ``env_type`` — which :class:`Environments` layout to build.
          * ``enable_dowham_reward_v1`` / ``enable_dowham_reward_v2`` /
            ``enable_count_based`` / ``enable_rnd`` — selects the intrinsic
            reward module (instantiated below). Typically only one is True.
          * ``max_steps`` — episode step budget (default 200).
          * ``size`` — grid side length (default 19); some layouts force 19.
          * ``tile_size`` — pixel size per tile for RGB rendering.
          * ``is_partial_obs`` / ``direction_obs`` / ``highlight`` /
            ``conv_filter`` — observation/rendering options.

        Any remaining ``kwargs`` are forwarded to ``EmptyEnv.__init__``.
        """
        self.shaped_reward = 0
        self.termination_reward = 0
        self.env_type = kwargs.pop("env_type", CustomEnv.Environments.empty)
        self.enable_dowham_reward_v1 = kwargs.pop('enable_dowham_reward_v1', False)
        self.enable_dowham_reward_v2 = kwargs.pop('enable_dowham_reward_v2', False)
        self.enable_count_based = kwargs.pop('enable_count_based', False)
        self.enable_rnd = kwargs.pop('enable_rnd', False)
        self.direction_obs = kwargs.pop('direction_obs', True)
        self.max_steps = kwargs.pop('max_steps', 200)
        self.conv_filter = kwargs.pop('conv_filter', False)
        self.is_partial_obs = kwargs.pop('is_partial_obs', True)
        self.highlight = kwargs.pop('highlight', False)
        self.percentage_visited = 0.0
        self.percentage_history = collections.deque(maxlen=100)
        self.action = None
        self.reward_range = (-1, 1)
        self.dowham_reward = None
        self.tile_size = kwargs.pop('tile_size', 12)
        self.see_through_walls = False
        self.size = kwargs.pop('size', 19)
        self.seed_sequence = np.random.SeedSequence()

        super().__init__(
            size=self.size,
            tile_size=self.tile_size,
            highlight=self.highlight,
            max_steps=self.max_steps,
            see_through_walls=False,
            agent_view_size=7,
            **kwargs)

        print(
            f"Custom Env {self.size}x{self.size} is used from {CustomEnv.__module__} with env type {self.env_type} and max steps {self.max_steps} \n"
            f"Intrinsic Rewards - DoWhaM V1: {self.enable_dowham_reward_v1}, \n"
            f"                    DoWhaM V2: {self.enable_dowham_reward_v2}, \n"
            f"                    Count Based: {self.enable_count_based}, \n"
            f"                    RND: {self.enable_rnd}")

        self.states = np.full((self.width, self.height), 0)
        self.reward_range = (-1, 1)
        self.is_intrinsic_reward_enabled = False
        # Later in the method, after other initializations
        if self.enable_rnd:
            print("RND Exploration Enabled")
            self.rnd = RNDModule(input_dim=148, embed_dim=32,  # Match fcnet_hiddens size
                                 hidden_size=32, reward_scale=1.0)
            self.is_intrinsic_reward_enabled = True
        if self.enable_dowham_reward_v1:
            print("Enable Dowham Reward V1")
            self.dowham_reward = DoWhaMIntrinsicRewardV1(eta=40, H=1, tau=0.5)
            self.is_intrinsic_reward_enabled = True
        if self.enable_dowham_reward_v2:
            print("Enable Dowham Reward V2")
            self.dowham_reward = DoWhaMIntrinsicRewardV2(eta=40, H=1, tau=0.5)
            self.is_intrinsic_reward_enabled = True
        if self.enable_count_based:
            print(f"Count Based Exploration Enabled")
            self.count_based = CountExploration(self, gamma=0.99, epsilon=0.1, alpha=0.1)
            self.is_intrinsic_reward_enabled = True

        print(f"Environment Type: {CustomEnv.Environments(self.env_type).name}")

        if self.env_type == CustomEnv.Environments.crossing:
            self.obstacle_type = Wall
            self.num_crossings = 1
            self.max_door = 1

        if self.env_type == CustomEnv.Environments.empty:
            self.max_door = 0

        if self.env_type == CustomEnv.Environments.four_rooms:
            self.max_door = 4

        if self.env_type == CustomEnv.Environments.multi_room:
            self.max_door = 3

        if self.env_type == CustomEnv.Environments.multi_room_key:
            self.max_door = 3

        self.intrinsic_reward = 0
        self.done = False

    @staticmethod
    def _gen_mission():
        """Return the default mission string shown to the agent."""
        return "get to the green goal square"

    def _reward(self) -> float:
        """Compute the success reward, decayed by how long the episode took.

        Returns a value in ``(1, 10]``: 10 for reaching the goal immediately,
        decaying linearly toward 1 as ``step_count`` approaches ``max_steps``.
        This rewards faster solutions.
        """
        return 10 - 9 * (self.step_count / self.max_steps)

    def transform_coords(self, x, y, agent_dir):
        """Rotate an East-facing ``(x, y)`` offset into the agent's facing frame.

        ``BASE_OFFSETS`` are defined assuming the agent faces East
        (``agent_dir == 0``). This applies the corresponding 90°-step rotation so
        the offset is expressed in global grid axes for the actual facing
        direction (0=right, 1=down, 2=left, 3=up). Returns the rotated
        ``(x, y)`` tuple, or ``None`` for an unrecognized direction.
        """
        if agent_dir == 0:  # Right (→), no change
            return x, y
        elif agent_dir == 1:  # Down (↓), rotate clockwise 90°
            return -y, x
        elif agent_dir == 2:  # Left (←), rotate 180°
            return -x, -y
        elif agent_dir == 3:  # Up (↑), rotate counter-clockwise 90°
            return y, -x
        return None

    def extract_visible_coords_from_obs(self, obs, agent_pos, agent_dir):
        """
        Map the agent's 7×7 egocentric type-mask to global coordinates.
        Includes all non-unseen, non-wall cells. Uses BASE_OFFSETS (East-facing)
        and transform_coords to rotate for any agent_dir.
        """

        # 1) Extract the raw type mask (shape VIEW_SIZE×VIEW_SIZE) and filter
        raw_img = obs['image'] if isinstance(obs, dict) else obs
        agent_col = 7 // 2
        agent_row = 7 - 1
        raw_img[agent_col, agent_row, 0] = OBJECT_TO_IDX['agent']
        mask = raw_img[:, :, 0].T  # Transpose to (height, width)
        valid = (mask != OBJECT_TO_IDX['unseen']) & (mask != OBJECT_TO_IDX['wall']) & (mask != OBJECT_TO_IDX['agent'])
        coords = np.argwhere(valid)

        # 2) Map each local (i,j) to global coords
        ax, ay = agent_pos
        visible = []
        for i, j in coords:
            # East-facing offset (dx,dy) from BASE_OFFSETS
            dx, dy = BASE_OFFSETS[i, j]
            # Rotate into actual direction
            rdx, rdy = self.transform_coords(dx, dy, agent_dir)
            wx, wy = ax + rdx, ay + rdy

            # Boundary check
            if 0 <= wx < self.width and 0 <= wy < self.height:
                visible.append((int(wx), int(wy)))

        return visible

    def step(self, action: int):
        """Advance the environment one step and shape the reward.

        Pipeline:
          1. Record the current cell into the visit grid and snapshot the
             pre-step observation hash, position, and direction.
          2. Delegate to ``super().step`` for the base MiniGrid transition.
          3. Compute the intrinsic reward with whichever method is enabled:
             - DoWhaM v1/v2: update visit/usage/effectiveness statistics, then
               score the action; v2 additionally uses the global coordinates of
               the cells visible before/after (via
               :meth:`extract_visible_coords_from_obs`) and the new position.
             - count-based: bonus from state-action visitation counts.
             - RND: prediction-error novelty bonus (see :meth:`rnd_reward`).
          4. Shape the reward: add the success bonus on termination and add the
             intrinsic reward scaled by 0.05; the total shaping is stored on
             ``self.shaped_reward`` and added to the env reward.

        Args:
            action: Discrete MiniGrid action index.

        Returns:
            The standard Gymnasium 5-tuple ``(obs, reward, terminated,
            truncated, info)`` with ``reward`` including the shaping above.
        """
        self.states[self.agent_pos[0]][self.agent_pos[1]] += 1
        self.action = action
        current_obs = self.gen_obs()["image"]
        current_obs_hash = self.hash_(current_obs)
        prev_pos = (self.agent_pos[0], self.agent_pos[1]) if isinstance(self.agent_pos, np.ndarray) else self.agent_pos
        prev_dir = self.agent_dir
        obs, reward, terminated, truncated, _ = super().step(action)
        next_obs_hash = self.hash_()
        next_obs = obs["image"]
        next_dir = self.agent_dir
        next_pos = (self.agent_pos[0], self.agent_pos[1]) if isinstance(self.agent_pos, np.ndarray) else self.agent_pos
        next_dir = self.agent_dir

        if self.enable_dowham_reward_v1 or self.enable_dowham_reward_v2:
            self.dowham_reward.update_state_visits(current_obs_hash, next_obs_hash)
            state_changed = current_obs_hash != next_obs_hash or prev_pos != next_pos
            self.dowham_reward.update_usage(current_obs_hash, action)

            self.dowham_reward.update_effectiveness(
                current_obs_hash,
                action,
                next_obs_hash,
                state_changed
            )

            if self.enable_dowham_reward_v2:
                curr_view = self.extract_visible_coords_from_obs(current_obs, prev_pos, prev_dir)
                next_view = self.extract_visible_coords_from_obs(next_obs, next_pos, next_dir)

                self.intrinsic_reward = self.dowham_reward.calculate_intrinsic_reward(
                    current_obs_hash,
                    action,
                    next_obs_hash,
                    state_changed,
                    curr_view,
                    next_view,
                    next_pos
                )
            else:
                self.intrinsic_reward = self.dowham_reward.calculate_intrinsic_reward(
                    current_obs_hash,
                    action,
                    next_obs_hash,
                    state_changed
                )

            # print(f"x:{self.agent_pos[0]}, y:{self.agent_pos[1]}, Intrinsic Reward: {self.intrinsic_reward}, Obs Hash: {current_obs_hash}, Next Obs Hash: {next_obs_hash}")

        if self.enable_count_based:
            self.intrinsic_reward = self.count_based.update((prev_pos[0], prev_pos[1]), action, reward, self.goal_pos)

        if self.enable_rnd:
            self.rnd_reward(obs)

        # --- Basic shaping: per-step penalty + success bonus, plus intrinsic on top ---
        self.shaped_reward = 0

        if terminated:
            self.shaped_reward += self._reward()
            self.termination_reward = self._reward()

        if self.is_intrinsic_reward_enabled:
            # Always add intrinsic reward (scaled), regardless of termination
            self.shaped_reward += self.intrinsic_reward * 0.05

        reward += self.shaped_reward

        self.done = terminated
        return obs, reward, terminated, truncated, {}

    def rnd_reward(self, obs):
        """Compute the RND novelty bonus for ``obs`` and update the predictor.

        Flattens the (possibly dict) observation into a float32 vector, updates
        the running observation normalizer, stores the prediction-error-based
        intrinsic reward on ``self.intrinsic_reward``, and takes a gradient step
        on the RND predictor network so already-seen states yield smaller bonuses
        over time.
        """
        # First, convert observation to a format suitable for RND
        if isinstance(obs, dict):
            # Flatten dict observation
            obs_list = []
            for key, value in obs.items():
                if isinstance(value, (int, np.int32, np.int64)):
                    value = np.array([value])
                elif not isinstance(value, np.ndarray):
                    value = np.array(value)
                obs_list.append(value.flatten())
            flat_obs = np.concatenate(obs_list)
        else:
            flat_obs = np.array(obs).flatten()

        flat_obs = flat_obs.astype(np.float32)
        self.rnd.update_obs_normalizer(flat_obs)
        # Calculate intrinsic reward
        self.intrinsic_reward = self.rnd.compute_intrinsic_reward(flat_obs)
        # Update predictor network
        self.rnd.update_predictor(flat_obs)

    def _gen_grid(self, width, height):
        """Dispatch to the layout builder selected by ``self.env_type``.

        Called by the base MiniGrid machinery during reset. Each branch
        populates ``self.grid``, the agent start, and the goal for one layout.
        """
        if self.env_type == CustomEnv.Environments.crossing:
            self.crossing_env(width, height)
        elif self.env_type == CustomEnv.Environments.empty:
            self.empty_env_random_goal(width, height)
        elif self.env_type == CustomEnv.Environments.four_rooms:
            self.four_rooms(width, height)
        elif self.env_type == CustomEnv.Environments.multi_room:
            self.multi_room(width, height)
        elif self.env_type == CustomEnv.Environments.multi_room_key:
            self.multi_room_key(width, height)
        elif self.env_type == CustomEnv.Environments.twelve_rooms:
            self.twelve_rooms(width, height)
        elif self.env_type == CustomEnv.Environments.long_corridor:
            self.long_corridor(width, height)

    def img_observation(self, size=32):
        """Return ``(state_hash, rgb_image)`` for the current state.

        For partial observability the agent's 7×7 egocentric image is used. For
        full observability a rendered RGB frame is produced; when
        ``direction_obs`` is False the frame is rendered with the agent
        temporarily forced to face East so orientation does not leak into the
        image. ``size`` controls the hash length.
        """
        if not self.is_partial_obs:
            if self.direction_obs:
                rgb_img = self.get_frame(
                    highlight=self.highlight, tile_size=self.tile_size
                )
            else:
                agent_dir = self.agent_dir
                self.agent_dir = 0
                rgb_img = self.get_frame(
                    highlight=self.highlight, tile_size=self.tile_size
                )
                self.agent_dir = agent_dir
        else:
            rgb_img = self.gen_obs()["image"]

        # Return the hashed value
        return self.hash_(size=size), rgb_img

    def hash_(self, current_obs=None, size=16):
        """Compute a short hash uniquely identifying the current state.

        Used as a dictionary key by the DoWhaM intrinsic-reward bookkeeping to
        recognize repeated states. If ``current_obs`` is given it is hashed
        directly; otherwise the agent's currently visible encoded grid is hashed.

        :param current_obs: Optional pre-computed observation image to hash.
        :param size: Number of leading hex digits of the digest to return.
        """
        sample_hash = hashlib.sha256()

        image = current_obs

        if current_obs is None:
            grid, vis_mask = self.gen_obs_grid()

            image = grid.encode(vis_mask)

        to_encode = [image.tolist(), ]

        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        return sample_hash.hexdigest()[:size]

    def crossing_env(self, width, height):
        """Build a SimpleCrossing-style layout with one wall "river" and a door.

        Walls form a barrier dividing the grid; a single opening (rendered as a
        closed yellow door) is carved so a path exists from the top-left agent
        start to the bottom-right goal. Requires odd ``width``/``height``.
        """
        import itertools as itt
        assert width % 2 == 1 and height % 2 == 1  # odd size
        self.obstacle_type = Wall
        self.num_crossings = 1

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        self.goal_pos = (width - 2, height - 2)

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[: self.num_crossings]  # sample random rivers
        rivers_v = sorted(pos for direction, pos in rivers if direction is v)
        rivers_h = sorted(pos for direction, pos in rivers if direction is h)
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1])
                )
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1])
                )
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, Door(color="yellow", is_open=False))
        self.max_door = 1
        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def empty_env_random_goal(self, width, height):
        """Build an empty walled room with the agent at (1,1) and goal at (17,17).

        Despite the name, the goal is currently fixed at (17, 17); the commented
        block shows how it was previously randomized.
        """
        self.max_door = 0
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        # Get grid size from environment
        grid_size = self.width  # Assuming width == height
        self.put_obj(Goal(), 17, 17)
        self.goal_pos = (17, 17)
        # Randomly assign a new goal position (excluding (1,1))
        # while True:
        #     self.goal_pos = (np.random.randint(1, grid_size - 2), np.random.randint(1, grid_size - 2))
        #     if self.goal_pos != (1, 1):  # Ensure it's not the starting position
        #         self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])
        #         break

    def four_rooms(self, width, height):
        """Build a classic FourRooms layout with doors at fixed wall midpoints.

        Two interior walls split the grid into four quadrants; four yellow doors
        (at the hard-coded midpoints) connect them. The agent starts top-left and
        the goal sits bottom-right.
        """
        self.max_door = 4
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0
        self.goal_pos = (width - 2, height - 2)
        self._agent_default_pos = self.agent_pos
        self._goal_default_pos = self.goal_pos
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    # self.grid.set(pos[0], pos[1], Door(color="yellow", is_open=False))

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    # self.grid.set(pos[0], pos[1], Door(color="yellow", is_open=False))
        self.grid.set(9, 4, Door(color="yellow", is_open=False))
        self.grid.set(9, 13, Door(color="yellow", is_open=False))
        self.grid.set(13, 9, Door(color="yellow", is_open=False))
        self.grid.set(4, 9, Door(color="yellow", is_open=False))
        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

    def multi_room(self, width, height):
        """
        Example of a 19×19 fixed, maze-like layout with several 'yellow' doors
        and a goal in the lower corridor.
        """
        # Force the grid size to 19×19.
        width = 19
        height = 19
        self.grid = Grid(width, height)

        # 1) Surrounding outer walls
        self.grid.wall_rect(1, 0, 5, 7)
        self.grid.wall_rect(0, 6, 7, 5)
        self.grid.wall_rect(6, 8, 9, 5)
        self.grid.wall_rect(14, 8, 5, 11)
        self.grid.set(3, 6, Door(color="yellow", is_open=False))
        self.grid.set(6, 9, Door(color="yellow", is_open=False))
        self.grid.set(14, 10, Door(color="yellow", is_open=False))
        self.max_door = 3
        # ----------------------------------------------------------------------
        # 4) Agent start in the top-left corridor
        # ----------------------------------------------------------------------
        self.agent_pos = (2, 1)
        self.agent_dir = 0  # facing right

        # ----------------------------------------------------------------------
        # 5) Goal in the lower corridor (the green square)
        # ----------------------------------------------------------------------
        self.put_obj(Goal(), 17, 17)
        self.goal_pos = (17, 17)

    def multi_room_key(self, width, height):
        """
        Example of a 19×19 fixed, maze-like layout with several locked doors
        requiring keys, and a goal in the lower corridor.
        """
        # Force the grid size to 19×19.
        width = 19
        height = 19
        self.grid = Grid(width, height)

        # 1) Surrounding outer walls
        self.grid.wall_rect(1, 0, 5, 7)
        self.grid.wall_rect(0, 6, 7, 5)
        self.grid.wall_rect(6, 8, 9, 5)
        self.grid.wall_rect(14, 8, 5, 11)
        self.grid.set(3, 6, Door(color="red", is_open=False, is_locked=True))
        self.grid.set(6, 9, Door(color="green", is_open=False, is_locked=True))
        self.grid.set(14, 10, Door(color="blue", is_open=False, is_locked=True))
        self.max_door = 3

        # Room definitions: (x_range, y_range, room_exit)
        rooms = {
            "1": ((2, 4), (1, 5), (3, 6)),
            "2": ((1, 5), (7, 9), (6, 9)),
            "3": ((7, 13), (9, 11), (14, 10)),
        }

        # Place one key per room: sample until we find a free cell
        for _, (x_range, y_range, room_exit) in rooms.items():
            min_x, max_x = x_range
            min_y, max_y = y_range

            door = self.grid.get(*room_exit)
            if not isinstance(door, Door):
                continue

            # Cap sampling attempts to avoid infinite loops on crowded rooms
            max_tries = (max_x - min_x + 1) * (max_y - min_y + 1) * 3
            tries = 0
            while tries < max_tries:
                key_pos = (
                    int(np.random.randint(min_x, max_x + 1)),
                    int(np.random.randint(min_y, max_y + 1)),
                )

                # Reject reserved cells and non-empty cells
                if (
                        key_pos != (2, 1)
                        and key_pos != (17, 17)
                        and self.grid.get(*key_pos) is None
                ):
                    self.put_obj(Key(color=door.color), *key_pos)
                    break
                tries += 1

        # self.put_obj(Key(color="red"), 2, 3)
        # self.put_obj(Key(color="green"), 5, 7)
        # self.put_obj(Key(color="blue"), 9, 11)
        # ----------------------------------------------------------------------
        # 4) Agent start in the top-left corridor
        # ----------------------------------------------------------------------
        self.agent_pos = (2, 1)
        self.agent_dir = 0  # facing right

        # ----------------------------------------------------------------------
        # 5) Goal in the lower corridor (the green square)
        # ----------------------------------------------------------------------
        self.put_obj(Goal(), 17, 17)
        self.goal_pos = (17, 17)

    def _multi_room(self, width, height):
        """Procedurally generate a connected chain of randomly placed rooms.

        Repeatedly calls :meth:`_placeRoom` to lay out non-overlapping rooms
        joined by colored doors (each door a different color from its
        predecessor), places the agent in the first room and the goal in the
        last. Unlike :meth:`multi_room`, this layout is randomized each reset.
        """
        self.minNumRooms = 2
        self.maxNumRooms = 2
        self.maxRoomSize = 9
        self.max_steps = self.maxNumRooms * 20

        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms + 1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (self._rand_int(0, width - 2), self._rand_int(0, width - 2))

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos,
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):
            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                self.grid.set(room.entryDoorPos[0], room.entryDoorPos[1], entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx - 1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = "traverse the rooms to get to the goal"

    def _placeRoom(self, numLeft, roomList, minSz, maxSz, entryDoorWall, entryDoorPos):
        """Recursively place one room and try to attach the remaining rooms.

        Picks a random room size anchored at ``entryDoorPos`` on the wall given
        by ``entryDoorWall`` (0=right, 1=south, 2=left, 3=top). Rejects positions
        that fall outside the grid or overlap existing rooms. On success the room
        is appended to ``roomList`` and, unless this was the last room
        (``numLeft == 1``), it attempts up to 8 exit-wall placements to recurse.

        Returns:
            bool: True if this room (and the recursion below it) was placed.
        """
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz + 1)
        sizeY = self._rand_int(minSz, maxSz + 1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = (
                    topX + sizeX < room.top[0]
                    or room.top[0] + room.size[0] <= topX
                    or topY + sizeY < room.top[1]
                    or room.top[1] + room.size[1] <= topY
            )

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(MultiRoom((topX, topY), (sizeX, sizeY), entryDoorPos, None))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):
            # Pick which wall to place the out door on
            wallSet = {0, 1, 2, 3}
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (topX + sizeX - 1, topY + self._rand_int(1, sizeY - 1))
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (topX + self._rand_int(1, sizeX - 1), topY + sizeY - 1)
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (topX, topY + self._rand_int(1, sizeY - 1))
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (topX + self._rand_int(1, sizeX - 1), topY)
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos,
            )

            if success:
                break

        return True

    def gen_obs(self):
        """Return the base MiniGrid observation as a shallow-copied dict."""
        obs = super().gen_obs()
        return {**obs}

    def reset(self, **kwargs):
        """Reset for a new episode and finalize the previous episode's stats.

        Before resetting, computes ``percentage_visited`` (fraction of cells
        with a nonzero visit count) for the episode just ended and pushes the
        terminal success flag onto ``percentage_history``. Then clears the visit
        grid, resets the active DoWhaM reward's per-episode state, restores the
        layout-appropriate agent start, and reseeds the base env with a fresh
        random seed so procedural layouts are re-randomized.
        """
        total_size = self.width * self.height
        # Calculate the number of unique states visited by the agent
        unique_states_visited = np.count_nonzero(self.states)

        # Calculate the percentage of the environment the agent has visited
        self.percentage_visited = (unique_states_visited / total_size) * 100
        self.percentage_history.append(self.done)
        self.states = np.full((self.width, self.height), 0)

        if self.enable_dowham_reward_v2 or self.enable_dowham_reward_v1:
            self.dowham_reward.reset_episode()

        if self.env_type == CustomEnv.Environments.multi_room:
            self.agent_pos = (2, 1)
        else:
            self.agent_pos = (1, 1)

        self.agent_dir = 0

        obs, _ = super().reset(**{**kwargs, "seed": np.random.randint(0, 2 ** 31 - 1, dtype=int)})
        self.intrinsic_reward = 0
        self.termination_reward = 0
        self.done = False
        return obs, {}

    def twelve_rooms(self, width, height):
        """Procedurally generate a 4-room layout behind locked, keyed doors.

        Like :meth:`_multi_room` but each connecting door is locked and a key of
        the matching color is scattered somewhere inside the room it leads out
        of, so the agent must collect keys to progress toward the goal in the
        final room.
        """
        self.minNumRooms = 4
        self.maxNumRooms = 4
        self.maxRoomSize = 12

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms + 1)
        roomList = []
        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (self._rand_int(0, width - 2), self._rand_int(0, width - 2))

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos,
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):
            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor, is_locked=True)
                self.grid.set(room.entryDoorPos[0], room.entryDoorPos[1], entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx - 1]
                prevRoom.exitDoorPos = room.entryDoorPos

        for idx, room in enumerate(roomList):
            topX, topY = room.top
            sizeX, sizeY = room.size
            while True:
                try:
                    key_pos = (np.random.choice(range(topX, topX + sizeX)),
                               np.random.choice(range(topY, topY + sizeY)))
                    obj = self.grid.get(key_pos[0], key_pos[1])
                    if self.agent_pos != key_pos and obj is None:  # Ensure it's not the starting position
                        x, y = room.exitDoorPos
                        door = self.grid.get(x, y)
                        self.put_obj(Key(color=door.color), key_pos[0], key_pos[1])
                        break
                except (ValueError, TypeError):
                    break
        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = "traverse the rooms to get to the goal"

    def long_corridor(self, width, height):
        """Build a "boredom trap" maze: a tempting dead-end loop plus a real exit.

        Fills the grid with walls, then carves a small start room, a colored
        rectangular loop adjacent to it (the distracting "boredom" path that
        leads nowhere), and a longer winding corridor that actually reaches the
        goal. Designed to probe whether an exploration bonus lures the agent into
        endlessly revisiting the loop instead of finding the exit.
        """
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        for x in range(1, width - 1):
            for y in range(1, height - 1):
                self.grid.set(x, y, Wall())

        start_w, start_h = 5, 5
        start_left = width // 2 - start_w // 2
        start_top = 1
        for x in range(start_left, start_left + start_w):
            for y in range(start_top, start_top + start_h):
                self.grid.set(x, y, None)

        self.agent_pos = (start_left + start_w // 2, start_top + start_h // 2)
        self.agent_dir = 1
        center_y = self.agent_pos[1]

        def build_path(points):
            coords = [points[0]]
            for i in range(len(points) - 1):
                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                assert x0 == x1 or y0 == y1, "Segments must be axis-aligned"
                dx = int(np.sign(x1 - x0))
                dy = int(np.sign(y1 - y0))
                steps = abs(x1 - x0) + abs(y1 - y0)
                cx, cy = x0, y0
                for _ in range(steps):
                    cx += dx
                    cy += dy
                    coords.append((cx, cy))
            return coords

        def carve_pattern(coords):
            seen = set()
            for x, y in coords:
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                color = "red" if (x + y) % 2 == 0 else "blue"
                self.grid.set(x, y, Floor(color=color))

        loop_right = start_left - 1
        loop_left = max(1, loop_right - 5)
        loop_top = start_top
        loop_bottom = start_top + start_h + 3
        if loop_left >= loop_right:
            raise ValueError("Grid too small for the boredom loop.")

        loop_points = [
            (loop_left, loop_top),
            (loop_right, loop_top),
            (loop_right, loop_bottom),
            (loop_left, loop_bottom),
            (loop_left, loop_top),
        ]
        loop_coords = build_path(loop_points)

        exit_entry_x = start_left + start_w
        if exit_entry_x + 2 >= width - 1:
            raise ValueError("Grid too small for the boredom exit.")

        exit_points = [
            (exit_entry_x, center_y),
            (width - 4, center_y),
            (width - 4, center_y + 5),
            (exit_entry_x + 2, center_y + 5),
            (exit_entry_x + 2, height - 4),
            (width - 3, height - 4),
            (width - 3, height - 2),
            (width - 2, height - 2),
        ]
        exit_coords = build_path(exit_points)

        carve_pattern(loop_coords + exit_coords)

        goal_pos = exit_coords[-1]
        self.grid.set(*goal_pos, None)
        self.put_obj(Goal(), *goal_pos)
        self.goal_pos = goal_pos
        self.mission = "avoid the boredom trap and find the goal"


def custom_trial_name(trial):
    """
    Creates a custom trial name based on the configuration.

    Args:
        trial: The trial object from Ray Tune

    Returns:
        str: Custom trial name including exploration type and environment type
    """
    env_config = trial.config.get("env_config", {})
    enable_dowham_reward_v1 = env_config.get("enable_dowham_reward_v1", False)
    enable_dowham_reward_v2 = env_config.get("enable_dowham_reward_v2", False)
    enable_count_based = env_config.get("enable_count_based", False)
    enable_rnd = env_config.get("enable_rnd", False)
    env_type = env_config.get("env_type", "unknown")
    train_batch_size = trial.config.get("train_batch_size", "unknown")
    fc = trial.config.get("model", {}).get("fcnet_hiddens", "unknown")
    grad_clip = trial.config.get("grad_clip", "unknown")

    exploration_type = "Default"
    if enable_dowham_reward_v1:
        exploration_type = "DoWhaMV1"
    elif enable_dowham_reward_v2:
        exploration_type = "DoWhaMV2"
    elif enable_count_based:
        exploration_type = "CountBased"
    elif enable_rnd:
        exploration_type = "RND"

    return f"{exploration_type}_{CustomEnv.Environments(env_type).name}_batch{train_batch_size}{fc}Dowham{enable_dowham_reward_v2}"


if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # CLI entry point. Parses arguments, initializes Ray, registers the env
    # under the chosen observation wrapper (conv/position/flat), builds a PPO
    # config, and launches either:
    #   * run_mode="experiment"            — a grid search over intrinsic-reward
    #     methods (DoWhaM v1, DoWhaM v2, count-based) × seeds, or
    #   * run_mode="hyperparameter_search" — a random search over PPO
    #     hyperparameters with no intrinsic reward.
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Custom training script")
    parser.add_argument('--num_rollout_workers', type=int, help='The number of rollout workers', default=1)
    parser.add_argument('--num_envs_per_worker', type=int, help='The number of environments per worker', default=1)
    parser.add_argument('--num_gpus', type=int, help='The number of GPUs to use', default=0)
    parser.add_argument('--num_samples', type=int, help='Number of samples', default=1)
    parser.add_argument('--verbose', type=int, help='Verbose log level', default=1)
    parser.add_argument('--timesteps_total', type=int, help='Timesteps Total', default=100_000_000)
    parser.add_argument('--max_steps', type=int, help='Max Time Steps', default=200)
    parser.add_argument('--num_cpus_per_env_runner', type=float, help='num_cpus_per_env_runner', default=0.25)
    parser.add_argument('--conv_filter', action="store_true", help='Use convolutional layer or flat observation')
    parser.add_argument('--evaluation_interval', type=int, default=3, help='Evaluation interval')
    parser.add_argument('--environment', type=str, help='Environment to choose', choices=[
        "empty",
        "crossing",
        "four_rooms",
        "multi_room",
        "multi_room_key",
        "long_corridor",
        "twelve_rooms",
    ], default="empty")
    parser.add_argument('--obs_type', type=str, choices=['conv', 'position', 'flat'], default='position',
                        help='Observation wrapper type: conv, position, or flat')
    parser.add_argument('--run_mode', type=str, choices=['experiment', 'hyperparameter_search'], required=True,
                        help='Specify whether to run an experiment or hyperparameter search')
    parser.add_argument('--trail_name', type=str, help='Custom trail name', default=None)
    parser.add_argument('--enable_dowham_reward_v1', action='store_true', )
    parser.add_argument('--enable_dowham_reward_v2', action='store_true', )
    parser.add_argument('--enable_count_based', action='store_true', )
    parser.add_argument('--enable_rnd', action='store_true', help='Enable RND exploration')
    args = parser.parse_args()

    print(f"\n Parsed arguments: {args} \n")

    ray.init(ignore_reinit_error=True, num_gpus=args.num_gpus, include_dashboard=False, log_to_driver=True,
             runtime_env={
                 "env_vars": {
                     "RAY_DISABLE_WORKER_STARTUP_LOGS": "0",
                     "RAY_LOG_TO_STDERR": "0",
                 }
             })

    env_type = CustomEnv.Environments[args.environment]

    if args.obs_type == 'conv':
        print("Observation wrapper type is conv")
        register_env("CustomPlaygroundCrossingEnv-v0",
                     lambda config:
                     ImgObsWrapper(RGBImgPartialObsWrapper(CustomEnv(**config), tile_size=12)))
        env = ImgObsWrapper(RGBImgPartialObsWrapper(CustomEnv(env_type=env_type), tile_size=12))
        obs = env.reset()
    elif args.obs_type == 'position':
        print("Observation wrapper type is position")
        register_env("CustomPlaygroundCrossingEnv-v0",
                     lambda config:
                     PositionBasedWrapper(CustomEnv(**config)))
        env = PositionBasedWrapper(CustomEnv(env_type=env_type))
        obs = env.reset()
    elif args.obs_type == 'flat':
        print("Observation wrapper type is flat")
        register_env("CustomPlaygroundCrossingEnv-v0",
                     lambda config:
                     FlatObsWrapper(FullyObsWrapper(CustomEnv(**config))))
        env = FlatObsWrapper(FullyObsWrapper(CustomEnv(env_type=env_type)))
        obs = env.reset()

    # Determine if any intrinsic motivation method is enabled
    no_intrinsic_motivation = not (args.enable_dowham_reward_v1 or args.enable_dowham_reward_v2 or
                                   args.enable_count_based or args.enable_rnd)

    # Base PPO configuration shared by both run modes. It is later deep-copied
    # and overlaid with per-trial overrides (env_config / hyperparameters). The
    # policy is an LSTM MLP (see model dict) trained with GAE, a KL penalty, and
    # a decaying entropy coefficient schedule.
    config = (
        PPOConfig()
        .training(
            use_critic=True,
            use_gae=True,
            use_kl_loss=True,
            kl_coeff=0.2,
            kl_target=0.01,
            vf_loss_coeff=0.5,
            entropy_coeff=0.006,
            train_batch_size_per_learner=16384,
            minibatch_size=2048,
            entropy_coeff_schedule=[
                [0, 0.006],
                [args.timesteps_total // 2, 0.002],
                [args.timesteps_total, 0.0],
            ],
            clip_param=0.3,
            vf_clip_param=10.0,
            lr_schedule=None,
            lr=2.5e-4,
            lambda_=0.95,
            gamma=0.99,
            num_epochs=6,
            model={'fcnet_hiddens': [512, 512],
                   'fcnet_activation': 'tanh',
                   'fcnet_weights_initializer': None,
                   'fcnet_weights_initializer_config': None,
                   'fcnet_bias_initializer': None,
                   'fcnet_bias_initializer_config': None,
                   'conv_activation': 'relu',
                   'conv_kernel_initializer': None,
                   'conv_kernel_initializer_config': None,
                   'conv_bias_initializer': None,
                   'conv_bias_initializer_config': None,
                   'conv_transpose_kernel_initializer': None,
                   'conv_transpose_kernel_initializer_config': None,
                   'conv_transpose_bias_initializer': None,
                   'conv_transpose_bias_initializer_config': None,
                   'post_fcnet_hiddens': [512],
                   'post_fcnet_activation': 'relu',
                   'post_fcnet_weights_initializer': None,
                   'post_fcnet_weights_initializer_config': None,
                   'post_fcnet_bias_initializer': None,
                   'post_fcnet_bias_initializer_config': None,
                   'free_log_std': False,
                   'log_std_clip_param': 20.0,
                   'no_final_linear': False,
                   'vf_share_layers': False,
                   'use_lstm': True,
                   'max_seq_len': 64,
                   'lstm_cell_size': 256,
                   'lstm_use_prev_action': True,
                   'lstm_use_prev_reward': True,
                   'lstm_weights_initializer': None,
                   'lstm_weights_initializer_config': None,
                   'lstm_bias_initializer': None,
                   'lstm_bias_initializer_config': None,
                   '_time_major': False,
                   'use_attention': False,
                   'attention_num_transformer_units': 1,
                   'attention_dim': 64,
                   'attention_num_heads': 1,
                   'attention_head_dim': 32,
                   'attention_memory_inference': 50,
                   'attention_memory_training': 50,
                   'attention_position_wise_mlp_dim': 32,
                   'attention_init_gru_gate_bias': 2.0,
                   'attention_use_n_prev_actions': 0,
                   'attention_use_n_prev_rewards': 0,
                   'framestack': False,
                   'dim': 88,
                   'grayscale': False,
                   'zero_mean': True,
                   'custom_model': None,
                   'custom_model_config': {},
                   'custom_action_dist': None,
                   'custom_preprocessor': None,
                   'encoder_latent_dim': None,
                   'always_check_shapes': False,
                   '_disable_preprocessor_api': False,
                   '_disable_action_flattening': False}

        ).learners(
            num_gpus_per_learner=0.8 if args.num_gpus > 0 else 0,
            num_learners=1,
            num_cpus_per_learner=1,
        )
        .experimental(
            _disable_preprocessor_api=True, )
        .environment(
            env="CustomPlaygroundCrossingEnv-v0",
            env_config={
                "enable_dowham_reward_v2": False,
                "env_type": env_type,
                "tile_size": 12,
                "max_steps": args.max_steps,
                "size": 19,
            },
            disable_env_checking=True,
            is_atari=False,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        .env_runners(
            num_env_runners=args.num_rollout_workers,
            num_envs_per_env_runner=args.num_envs_per_worker,
            num_cpus_per_env_runner=args.num_cpus_per_env_runner,
            num_gpus_per_env_runner=0,
            batch_mode="truncate_episodes",
            rollout_fragment_length=64,
        )
        .framework("torch")
        .debugging(
            fake_sampler=False,
        ).api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        ).callbacks(CustomCallback)
        .evaluation(
            evaluation_interval=args.evaluation_interval,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=False,
            evaluation_sample_timeout_s=120,
        ).fault_tolerance(
            restart_failed_env_runners=True
        ).reporting(
            min_time_s_per_iteration=30
        )
    )

    algo = config.build_algo()

    # Keep the 5 best checkpoints by shortest mean episode length
    checkpoint_config = CheckpointConfig(
        num_to_keep=5,
        checkpoint_frequency=5,
        checkpoint_at_end=True,
        checkpoint_score_attribute="env_runners/episode_len_mean",
        checkpoint_score_order="min"
    )

    if args.run_mode == 'experiment':
        # Grid-search one trial per intrinsic-reward method × seed, all on the
        # chosen environment, minimizing mean episode length.
        search_space = {
            **copy.deepcopy(config),
            "env_config": tune.grid_search([
                # Default PPO
                {
                    "env_type": CustomEnv.Environments[args.environment],
                    "max_steps": args.max_steps,
                    "conv_filter": args.conv_filter,
                    "enable_dowham_reward_v1": True,
                    "enable_dowham_reward_v2": False,
                    "enable_count_based": False,
                    "enable_rnd": False,
                },
                {
                    "env_type": CustomEnv.Environments[args.environment],
                    "max_steps": args.max_steps,
                    "conv_filter": args.conv_filter,
                    "enable_dowham_reward_v1": False,
                    "enable_dowham_reward_v2": True,
                    "enable_count_based": False,
                    "enable_rnd": False,
                },
                {
                    "env_type": CustomEnv.Environments[args.environment],
                    "max_steps": args.max_steps,
                    "conv_filter": args.conv_filter,
                    "enable_dowham_reward_v1": False,
                    "enable_dowham_reward_v2": False,
                    "enable_count_based": True,
                    "enable_rnd": False,
                },
            ]),
            "seed": tune.grid_search(list(range(args.num_samples))),
        }

        trail = tune.run(
            "PPO",  # Specify the RLlib algorithm
            config=search_space,
            metric="env_runners/episode_len_mean",
            mode="min",
            stop={
                "timesteps_total": args.timesteps_total,
            },
            checkpoint_config=checkpoint_config,
            trial_name_creator=custom_trial_name,  # Custom trial name
            verbose=args.verbose,  # Display detailed logs
            num_samples=1,
            log_to_file=True,
            resume=True,
            max_failures=-1,
            reuse_actors=False,
            name=args.trail_name if hasattr(args, 'trail_name') and args.trail_name else None
        )
    elif args.run_mode == 'hyperparameter_search':
        # Random search (BasicVariantGenerator) over core PPO hyperparameters
        # with no intrinsic reward; the best config by mean episode length is
        # printed at the end.
        trail = tune.Tuner(
            "PPO",  # Specify the RLlib algorithm
            param_space={
                **copy.deepcopy(config),
                "env_config": {
                    "enable_dowham_reward_v2": False,
                    "enable_dowham_reward_v1": False,
                    "enable_count_based": False,
                    "enable_rnd": False,
                    "env_type": CustomEnv.Environments[args.environment],
                    "max_steps": 1444,
                },
                "gamma": tune.choice([0.99, 0.9997]),
                "lambda_": tune.uniform(0.95, 0.99),  # GAE lambda
                "train_batch_size": tune.choice([1024, 2048, 4096]),  # Batch size for training
                "num_epochs": tune.choice([10, 20, 30]),  # Number of epochs per training iteration
                "vf_loss_coeff": tune.uniform(0.1, 1.0),  # Value function loss coefficient
                "entropy_coeff": tune.loguniform(5e-4, 2e-2),  # Entropy coefficient for exploration

                # === EXPLORATION AND REGULARIZATION ===
                # "entropy_coeff": tune.loguniform(1e-4, 5e-2),  # Encourage exploration
                # "vf_loss_coeff": tune.uniform(0.1, 1.0),  # Value function importance
                # "grad_clip": tune.uniform(0.5, 10.0),  # Gradient clipping for stability
            },
            tune_config=tune.TuneConfig(
                metric="env_runners/episode_len_mean",
                mode="min",
                num_samples=args.num_samples,
                reuse_actors=False,
                search_alg=BasicVariantGenerator(),
                # Use Bayesian optimization
            ),
            run_config=train.RunConfig(stop={"timesteps_total": args.timesteps_total},
                                       failure_config=FailureConfig(max_failures=-1)),
        )

        results = trail.fit()
        best_trial = results.get_best_result(metric="env_runners/episode_len_mean", mode="min")
        print("Best Hyperparameters:", best_trial.config)
