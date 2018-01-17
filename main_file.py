#!/usr/bin/env python
from grid_world import GridWorld
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np
from operator import add


def main():
    wall_locs = [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)]
    gridworld = GridWorld(height=6, width=9, start_state=(2, 0),
                          goal_state=(0, 8), wall_locations=wall_locs)

    agents = [
        Agent(gridworld, n=0),
        Agent(gridworld, n=5),
        Agent(gridworld, n=50)
        ]

    episodes = 50

    total_steps = []
    for agent in agents:
        steps = [0] * episodes
        for each_run in range(30):
            num_steps = []
            for ep in range(episodes):
                actions, total_reward, state_action_list = agent.run_episode()
                num_steps.append(len(actions))
            steps = map(add, num_steps, steps)
            agent.reset()
        total_steps.append([i/30 for i in steps])

    agent_labels = [
        'n = 0',
        'n = 5',
        'n = 50'
    ]
    for i, num_steps in enumerate(total_steps):
        plt.plot(range(1, episodes+1), num_steps, label=agent_labels[i])
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode')
    plt.title('Dyna-Q Results')
    plt.show()


if __name__ == "__main__":
    main()
