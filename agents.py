import numpy as np
import random
import pickle

class PacmanAgent_QLearning:
    def __init__(self, env):
        self.env = env
        self.total_reward = 0
        self.state = None

        self.action_space = env.action_space
        self.state_space = env.pacman_state_space

        self.Q = np.zeros([self.state_space, len(list(self.action_space.values()))])

    def init_episode(self, env):
        self.env = env
        self.total_reward = 0
        self.action_space = env.action_space
        self.state_space = env.pacman_state_space

    def step_optimal_policy(self):
        (state, state_index) = self.env.get_pacman_state()
        action = np.argmax(self.Q[state_index]) # Exploit learned values

        (next_state, next_state_index) , reward , done , info = self.env.pacmam_step(action)

        # Update total reward
        self.total_reward += reward

        return reward, done, info

    def step_q_learning(self, gamma=0.7, alpha=0.2, epsilon=0.1):
        (state, state_index) = self.env.get_pacman_state()

        done = False

        if random.uniform(0, 1) < epsilon:
            action = random.choice(list(self.action_space.values())) # Explore state space
        else:
            action = np.argmax(self.Q[state_index]) # Exploit learned values

        (next_state, next_state_index) , reward , done , info = self.env.pacmam_step(action)

        # Update total reward
        self.total_reward += reward

        # Update Q table
        next_max = np.max(self.Q[next_state_index])
        old_value = self.Q[state_index, action]
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        self.Q[state_index ,action] = new_value

        return reward, done, info

    def save_q_table(self, version_name='output/'):
        npy_path = version_name+'_pacman_q_table.npy'
        with open(npy_path, 'wb') as f:
            np.save(f, self.Q)

class PacmanAgent_Cheating_QLearning:
    def __init__(self, env):
        self.env = env
        self.total_reward = 0
        self.state = None

        self.action_space = env.action_space
        self.state_space = env.ghost_state_space

        self.Q = np.zeros([self.state_space, len(list(self.action_space.values()))])

    def init_episode(self, env):
        self.env = env
        self.total_reward = 0
        self.action_space = env.action_space
        self.state_space = env.ghost_state_space

    def step_optimal_policy(self):
        (state, state_index) = self.env.get_ghost_state(0)
        action = np.argmax(self.Q[state_index]) # Exploit learned values

        (next_state, next_state_index) , reward , done , info = self.env.pacmam_step(action)

        # Update total reward
        self.total_reward += reward

        return reward, done, info

    def step_q_learning(self, gamma=0.7, alpha=0.2, epsilon=0.1):
        (state, state_index) = self.env.get_ghost_state(0)

        done = False

        if random.uniform(0, 1) < epsilon:
            action = random.choice(list(self.action_space.values())) # Explore state space
        else:
            action = np.argmax(self.Q[state_index]) # Exploit learned values

        (next_state, next_state_index) , reward , done , info = self.env.pacmam_step(action)

        (next_state, next_state_index) = self.env.get_ghost_state(0)

        # Update total reward
        self.total_reward += reward

        # Update Q table
        next_max = np.max(self.Q[next_state_index])
        old_value = self.Q[state_index, action]
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        self.Q[state_index ,action] = new_value

        return reward, done, info

    def save_q_table(self, version_name='output/'):
        npy_path = version_name+'_pacman_cheating_q_table.npy'
        with open(npy_path, 'wb') as f:
            np.save(f, self.Q)


class GhostAgent_Random:
    def __init__(self, env, ghost_index):
        self.env = env
        self.ghost_index = ghost_index
        self.total_reward = 0

        self.action_space = env.action_space

    def init_episode(self, env):
        self.env = env
        self.total_reward = 0
        self.action_space = env.action_space

    def step_random(self):
        # Select random action
        action = random.choice(list(self.action_space.values()))

        # Take that action
        (next_state, state_value) , reward , done , info = self.env.ghost_step(self.ghost_index, action)

        self.total_reward += reward
        return reward, done, info