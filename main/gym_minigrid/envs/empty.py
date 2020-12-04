from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from ray.tune.registry import register_env

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=7,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        agent_view_size = 5,
        env_config = None
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(3)

        # Number of cells (width and height) in the agent view
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        # self.observation_space = spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(5, 5, 3),
        #     # shape=(self.agent_view_size, self.agent_view_size, 3),
        #     dtype='uint8'
        # )

        super().__init__(
            grid_size=size,
            max_steps=size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

class EmptyEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5)

class EmptyRandomEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class EmptyEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6)

class EmptyRandomEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)

class EmptyEnv16x16(EmptyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-Empty-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5'
)

def EmptyEnv5x5Creator(env_config):
    return EmptyEnv(size=5)

register_env('MiniGrid-Empty-5x5-v0', EmptyEnv5x5Creator)


# register(
#     id='MiniGrid-Empty-Random-5x5-v0',
#     entry_point='gym_minigrid.envs:EmptyRandomEnv5x5'
# )

# def EmptyEnv5x5RanCreator(env_config):
#     return EmptyEnv(size=5)

# register_env('MiniGrid-Empty-5x5-v0', EmptyEnv5x5Creator)



# register(
#     id='MiniGrid-Empty-6x6-v0',
#     entry_point='gym_minigrid.envs:EmptyEnv6x6'
# )

# register(
#     id='MiniGrid-Empty-Random-6x6-v0',
#     entry_point='gym_minigrid.envs:EmptyRandomEnv6x6'
# )

# register(
#     id='MiniGrid-Empty-8x8-v0',
#     entry_point='gym_minigrid.envs:EmptyEnv'
# )

# register(
#     id='MiniGrid-Empty-16x16-v0',
#     entry_point='gym_minigrid.envs:EmptyEnv16x16'
# )
