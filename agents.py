# import imp
import numpy as np
import random
import pickle
from utils import q_table_dict
import json

class PacmanAgent_QLearning:
    def __init__(self, env):
        self.env = env

        self.action_space = env.action_space
        self.Q = q_table_dict(self.action_space.n)

    def predict(self, state, deterministic=True, epsilon=0.1):

        action = self.Q.get_max_action(state)
        return action, None

    def step_q_learning(self, gamma=0.7, alpha=0.2, epsilon=0.1):
        state = self.env.get_pacman_state()

        done = False

        if random.uniform(0, 1) < epsilon:
            action = random.choice(list(range(self.action_space.n))) # Explore state space
        else:
            # action = np.argmax(self.Q[state_index]) # Exploit learned values
            action = self.Q.get_max_action(state)

        next_state, reward, done, info = self.env.step(action)

        # Update Q table
        next_max = self.Q.get_max_value(next_state)
        old_value = self.Q.get_state_action(state, action)
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)

        self.Q.set_state_action(state, action, new_value)

        return next_state, reward, done, info

    def save_q_table(self, version_name='output/'):
        var = dict()
        var['q_table'] = list(self.Q.data.items())
        with open(version_name+'pacman_q_table.json', 'w') as outfile:
            json.dump(var, outfile)
        # npy_path = version_name+'_pacman_q_table.npy'
        # with open(npy_path, 'wb') as f:
        #     np.save(f, self.Q)


class GhostAgent_Random:
    def __init__(self, env, ghost_index):
        self.env = env
        self.ghost_index = ghost_index

        self.action_space = env.action_space


    def step_random(self):
        # Select random action
        action = random.choice(list(range(self.action_space.n)))

        # Take that action
        next_state, reward , done , info = self.env.ghost_step(self.ghost_index, action)

        return reward, done, info

    def predict(self, state, deterministic=True, epsilon=0.1):
        return random.choice(list(range(self.action_space.n))), None

class GhostAgent_QLearning:
    def __init__(self, env, ghost_index):
        self.env = env
        self.ghost_index = ghost_index

        self.action_space = env.action_space
        self.Q = q_table_dict(self.action_space.n)

    def predict(self, state, deterministic=True, epsilon=0.1):

        action = self.Q.get_max_action(state)
        return action, None

    def step_q_learning(self, gamma=0.7, alpha=0.2, epsilon=0.1):
        state = self.env.get_ghost_state(1)

        done = False

        if random.uniform(0, 1) < epsilon:
            action = random.choice(list(range(self.action_space.n))) # Explore state space
        else:
            # action = np.argmax(self.Q[state_index]) # Exploit learned values
            action = self.Q.get_max_action(state)

        next_state, reward, done, info = self.env.step(action)

        # Update Q table
        next_max = self.Q.get_max_value(next_state)
        old_value = self.Q.get_state_action(state, action)
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)

        self.Q.set_state_action(state, action, new_value)

        return next_state, reward, done, info

    def save_q_table(self, version_name='output/'):
        var = dict()
        var['q_table'] = list(self.Q.data.items())
        with open(version_name+'pacman_q_table.json', 'w') as outfile:
            json.dump(var, outfile)
        # npy_path = version_name+'_pacman_q_table.npy'
        # with open(npy_path, 'wb') as f:
        #     np.save(f, self.Q)