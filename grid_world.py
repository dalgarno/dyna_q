import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


class GridWorld(object):
    """
    DOCSTRING
    """

    def __init__(self, height=5, width=5, start_state=(0, 0), goal_state=(1,1),
                 wall_locations=None):
        super(GridWorld, self).__init__()
        self.height = height
        self.width = width
        self.start_state = start_state
        self.goal_state = goal_state
        self.wall_locations = wall_locations
        self.gridworld = self.create_gridworld(self.wall_locations)

    def __repr__(self):
        return '{}\n{}'.format(self.__class__.__name__, self.gridworld)

    def create_gridworld(self, walls):
        grid = np.zeros((self.height, self.width), dtype=np.int)

        # Set wall values
        if walls:
            for wall in walls:
                grid[wall] = 1

        # set start location
        grid[self.start_state] = 2

        # set goal location
        grid[self.goal_state] = 2

        return grid

    def view_gridworld(self):
        fig, ax = plt.subplots()
        ax.imshow(self.gridworld, cmap='Greys')
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(-0.5, self.width, 1))
        ax.set_yticks(np.arange(-0.5, self.height, 1))
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticklabels([])
        cur_axes.axes.get_yaxis().set_ticklabels([])

        plt.title('Gridworld')
        plt.show()

    def rearrange_gridworld(self, new_walls):
        self.gridworld = self.create_gridworld(new_walls)

    def available_states(self):
        return list(zip(*np.where(self.gridworld != 1)))
