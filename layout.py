# layout.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from utils import Grid

import random
import copy

class Env:
    """
    A Layout manages the static information about the game board.
    """

    def __init__(self, layoutText):
        # self.see_food = 1
        self.layoutText = layoutText
        self.width = len(layoutText[0])
        self.height= len(layoutText)
        self.walls = Grid(self.width, self.height, False)
        self.food = Grid(self.width, self.height, False)
        self.food_possible_positions = []
        self.pacman_position = (None, None)
        self.ghost_positions = []

        self.processLayoutText(layoutText)          # Fill out walls, food, food_possible_positions
                                                    #          pacman_position, ghost_positions

        self.action_space = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        # self.pacman_state_space = self.width * self.height * (2**self.see_food)
        self.ghost_state_space = self.width * self.height * self.width * self.height * (2**len(self.food_possible_positions))

        self.init_transition_matrix()




    ####################################
    ##        Callable Functions      ##
    ####################################

    def pacmam_step(self, action_value):
        cur_x, cur_y = self.pacman_position
        next_x, next_y = self.transition_matrix[cur_y][cur_x][action_value]

        # Update pacman position in ENV
        self.pacman_position = (next_x, next_y)

        info = ''

        # Calculate reward and check if done:
        reward = -1
        done = False

        # Check meeting food
        for (food_x, food_y) in self.food_possible_positions:
            if next_x == food_x and next_y == food_y:
                if self.food[next_y][next_x]:
                    self.food[next_y][next_x] = False
                    reward += 300
                    info += '\tmeet food at ({},{})\n'.format(next_x, next_y)
        
        # Check no food left
        if len(self.food.asList()) == 0:
            reward += 1000
            done = True
            info += '\tDone because no more food\n'

        # Check meeting ghost
        for (ghost_x, ghost_y) in self.ghost_positions:
            if next_x == ghost_x and next_y == ghost_y:
                reward -= 300
                done = True
                info += '\tDone because meeting ghost at ({},{})\n'.format(next_x, next_y)

        # Construct next state
        next_state = self.get_pacman_state()

        return (next_state, reward, done, info)


    def ghost_step(self, ghost_index, action_value):
        cur_x, cur_y = self.ghost_positions[ghost_index]
        next_x, next_y = self.transition_matrix[cur_y][cur_x][action_value]

        # Update ghost position in ENV
        self.ghost_positions[ghost_index] = (next_x, next_y)

        info = ''

        # Calculate reward and check if done:
        reward = -1
        done = False

        # Check meeting pacman
        pacman_x, pacman_y = self.pacman_position
        if next_x == pacman_x and next_y == pacman_y:
            reward += 500
            done = True
            info += '\tDone because meeting pacman at ({},{})\n'.format(next_x, next_y)

        # Construct next state
        next_state = self.get_ghost_state(ghost_index)

        return (next_state, reward, done, info)


    def get_pacman_state(self):
        # Observe if food in possible positions
        food_binary_list = []
        for i, (food_x, food_y) in enumerate(self.food_possible_positions):
            # if i >= self.see_food:
            #     break
            if self.food[food_y][food_x]:
                food_binary_list.append(1)
            else:
                food_binary_list.append(0)

        state = (self.pacman_position, tuple(self.ghost_positions), tuple(self.food_possible_positions), tuple(food_binary_list))
        state_value = self.pacman_position[0]
        state_value *= self.width
        state_value =+ self.pacman_position[1]

        if len(food_binary_list) > 0:
            state_value *= self.height
            state_value += food_binary_list[0]
            if len(food_binary_list) > 1:
                for item in food_binary_list[1:]:
                    state_value *= 2
                    state_value += item

        return (state, state_value)

    def get_ghost_state(self, ghost_index):
        # Observe if food in possible positions
        food_binary_list = []
        for (food_x, food_y) in self.food_possible_positions:
            if self.food[food_y][food_x]:
                food_binary_list.append(1)
            else:
                food_binary_list.append(0)

        state = [self.ghost_positions[ghost_index], self.pacman_position, food_binary_list]
        state_value = self.ghost_positions[ghost_index][0]
        state_value *= self.width
        state_value =+ self.ghost_positions[ghost_index][1]
        state_value *= self.height
        state_value += self.pacman_position[0]
        state_value *= self.width
        state_value =+ self.pacman_position[1]

        if len(food_binary_list) > 0:
            state_value *= self.height
            state_value += food_binary_list[0]
            if len(food_binary_list) > 1:
                for item in food_binary_list[1:]:
                    state_value *= 2
                    state_value += item

        return (state, state_value)


    def deep_copy_new(self):
        return Env(self.layoutText[:])

    ####################################
    ##        Internal Functions      ##
    ####################################

    def init_transition_matrix(self):
        self.transition_matrix = []
        action_values = self.action_space.values()
        for y in range(self.height):
            cur_row_transition = []
            for x in range(self.width):
                cur_pos_transition = dict()
                for action_value in action_values:
                    next_x, next_y = self.get_next_position(x, y, action_value)
                    cur_pos_transition[action_value] = (next_x, next_y)
                cur_row_transition.append(cur_pos_transition)
            self.transition_matrix.append(cur_row_transition)

    def get_next_position(self, cur_x, cur_y, action_value):
        next_x, next_y = None, None

        if action_value == 0:                                       # Goes up
            if cur_y == 0 or self.isWall(cur_x, cur_y-1):
                next_x, next_y = cur_x, cur_y
            else:
                next_x, next_y = cur_x, cur_y-1
        elif action_value == 1:                                     # Goes down
            if cur_y == self.height-1 or self.isWall(cur_x, cur_y+1):
                next_x, next_y = cur_x, cur_y
            else:
                next_x, next_y = cur_x, cur_y+1
        elif action_value == 2:                                     # Goes left
            if cur_x == 0 or self.isWall(cur_x-1, cur_y):
                next_x, next_y = cur_x, cur_y
            else:
                next_x, next_y = cur_x-1, cur_y
        elif action_value == 3:                                     # Goes right
            if cur_x == self.width-1 or self.isWall(cur_x+1, cur_y):
                next_x, next_y = cur_x, cur_y
            else:
                next_x, next_y = cur_x+1, cur_y
        elif action_value == 4:
            next_x, next_y = cur_x, cur_y
        else:
            print('ERROR: invalid action: {} at position {}'.format(action_value, (cur_x, cur_y)))

        return next_x, next_y

    def isWall(self, x, y):
        return self.walls[y][x]

    def getRandomLegalPosition(self):
        x = random.choice(range(self.width))
        y = random.choice(range(self.height))
        while self.isWall(x, y):
            x = random.choice(range(self.width))
            y = random.choice(range(self.height))
        return (x,y)

    def __str__(self):
        return "\n".join(self.layoutText)

    def processLayoutText(self, layoutText):
        """
        Coordinates are flipped from the input format to the (x,y) convention here

        The shape of the maze.  Each character
        represents a different type of object.
         % - Wall
         . - Food
         G - Ghost
         P - Pacman
        Other characters are ignored.
        """
        for y in range(self.height):
            for x in range(self.width):
                layoutChar = layoutText[y][x]
                self.processLayoutChar(x, y, layoutChar)

    def processLayoutChar(self, x, y, layoutChar):
        if layoutChar == '%':
            self.walls[y][x] = True
        elif layoutChar == '.':
            self.food[y][x] = True
            self.food_possible_positions.append((x, y))
        elif layoutChar == 'O':
            self.food_possible_positions.append((x, y))
        elif layoutChar == 'P':
            self.pacman_position = (x, y)
        elif layoutChar == 'G':
            self.ghost_positions.append((x, y))


def load_env(layout_path):

    with open(layout_path, 'r') as f:
        lines = [line.strip() for line in f]

    return Env(lines)