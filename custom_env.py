from minigrid.core.world_object import Door, Goal
from minigrid.envs import MultiRoomEnv

from dowham import DoWhaMIntrinsicReward


class CustomMultiRoomEnv(MultiRoomEnv):
    def __init__(self, *args, **kwargs):
        self.max_episode_steps = kwargs.pop('max_episode_steps', 100)
        self.enable_dowham = kwargs.pop('enable_dowham', False)
        self.dowham_reward = kwargs.pop('dowham_reward', DoWhaMIntrinsicReward(eta=1.5, H=1.0, tau=0.5))
        # Extract custom configurations or set defaults
        self.previous_distance = 0
        self.custom_agent_start_pos = kwargs.pop('agent_start_pos', (1, 1))
        self.custom_agent_start_dir = kwargs.pop('agent_start_dir', 0)  # 0: right, 1: down, 2: left, 3: up
        self.custom_goal_pos = kwargs.pop('goal_pos', (7, 7))

        self.agent_pos = self.custom_agent_start_pos
        self.agent_dir = self.custom_agent_start_dir

        super().__init__(*args, **kwargs)

    def _gen_grid(self, width, height):
        # Create the grid and place walls around the border
        self.grid.wall_rect(0, 0, width, height)

        # Place a vertical wall to divide the grid into two halves
        self.grid.vert_wall(width // 2, 0, height)

        # Place a horizontal wall to divide the grid into two halves
        self.grid.horz_wall(0, height // 2, width)

        # Place a door in each wall to connect the rooms
        self.put_obj(Door('yellow'), width // 2, height // 4)  # Door in the upper part of the vertical wall
        self.put_obj(Door('yellow'), width // 2, 3 * height // 4)  # Door in the lower part of the vertical wall
        self.put_obj(Door('yellow'), width // 4, height // 2)  # Door in the left part of the horizontal wall
        self.put_obj(Door('yellow'), 3 * width // 4, height // 2)  # Door in the right part of the horizontal wall

        # Place the goal in the center room
        self.put_obj(Goal(), self.custom_goal_pos[0], self.custom_goal_pos[1])

        # Set agent's start position and direction based on custom configuration
        self.agent_pos = self.custom_agent_start_pos
        self.agent_dir = self.custom_agent_start_dir if self.custom_agent_start_dir is not None else 0

    def step(self, action):
        current_state = self.agent_pos
        obs, reward, done, info, _ = super().step(action)

        if self.enable_dowham:
            next_state = self.agent_pos
            self.dowham_reward.update_state_visits(next_state)
            state_changed = current_state != next_state
            self.dowham_reward.update_usage(action)
            self.dowham_reward.update_effectiveness(action, state_changed)
            intrinsic_reward = self.dowham_reward.calculate_intrinsic_reward(action, current_state, next_state)
            reward += intrinsic_reward

        return obs, reward, done, info, {}

#
#
# class CustomFlatObsWrapper(FlatObsWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#
# if __name__ == "__main__":
#     env_config = {"render_mode": "human"}
#     env = CustomMultiRoomEnv(agent_start_pos=(1, 1),
#                              agent_start_dir=0,
#                              goal_pos=(18, 18),
#                              minNumRooms=2,
#                              maxNumRooms=5, **env_config)
#     env.reset()
#     env = CustomFlatObsWrapper(env)
#     env.reset()
#     while True:
#         env.render()
