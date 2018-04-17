import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import clear_output


class MountainCar:
    def __init__(self, position_min=-1.2, position_max=0.5, velocity_min=-0.07, velocity_max=0.07):
        self.position_min = position_min
        self.position_max = position_max
        self.velocity_min = velocity_min
        self.velocity_max = velocity_max

        self.actions = np.array([-1, 0, 1])
        self.reset()

    def reset(self):
        self.game_over = False
        self.position = np.random.uniform(low=self.position_min, high=self.position_max)
        self.velocity = np.random.uniform(low=self.velocity_min, high=self.velocity_max)

    def make_step(self, action):
        """Given an action, move the car.

        Makes sure that position and velocity stay within specified bounds.
        Updates self.game_over if the right position bound is reached.

        @param action: an integer representing one of three actions
            -1 = full throttle reverse
             0 = zero throttle
             1 = full throttle forward

        @return reward: a float always valuing -1

        """
        if action not in self.actions:
            raise ValueError("The action value (", action, ") should be one of", self.actions)
        self.game_over = False
        self.velocity = self.velocity + 0.001 * action - 0.0025 * np.cos(3 * self.position)
        self.velocity = np.clip(self.velocity, a_min=self.velocity_min, a_max=self.velocity_max)

        self.position = self.position + self.velocity
        self.position = np.clip(self.position, a_min=self.position_min, a_max=self.position_max)
        if self.position == self.position_min:
            self.velocity = 0
        if self.position == self.position_max:
            self.game_over = True
        reward = -1.0
        return reward

    def plot(self, clear_the_output=True):
        if clear_the_output:
            clear_output(wait=True)

        fig, ax = plt.subplots()
        ax.axis([self.position_min, self.position_max, -1.1, 1.1])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)  # disable the grid
        xes = np.linspace(start=-1.2, stop=0.5, num=120)
        ys = np.sin(3 * xes)
        # Plot mountain
        ax.plot(xes, ys)
        # Plot "car"
        p = mpatches.Circle((self.position, np.sin(3 * self.position)), 0.05, color="red")
        ax.add_patch(p)
        plt.show()

