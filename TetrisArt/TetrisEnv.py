#from __future__ import print_function

import numpy as np
import random
import gym
from typing import Tuple, Optional

shapes = {
    'T': [(0, 0), (-1, 0), (1, 0), (0, -1)],
    'J': [(0, 0), (-1, 0), (0, -1), (0, -2)],
    'L': [(0, 0), (1, 0), (0, -1), (0, -2)],
    'Z': [(0, 0), (-1, 0), (0, -1), (1, -1)],
    'S': [(0, 0), (-1, -1), (0, -1), (1, 0)],
    'I': [(0, 0), (0, -1), (0, -2), (0, -3)],
    'O': [(0, 0), (0, -1), (-1, 0), (-1, -1)],
}
shape_names = ['T', 'J', 'L', 'Z', 'S', 'I', 'O']


def rotated(shape, cclk=False):
    if cclk:
        return [(-j, i) for i, j in shape]
    else:
        return [(j, -i) for i, j in shape]


def is_occupied(shape, anchor, board):
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        if y < 0:
            continue
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
            return True
    return False


def left(shape, anchor, board):
    new_anchor = (anchor[0] - 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def right(shape, anchor, board):
    new_anchor = (anchor[0] + 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def soft_drop(shape, anchor, board):
    new_anchor = (anchor[0], anchor[1] + 1)
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def hard_drop(shape, anchor, board):
    while True:
        _, anchor_new = soft_drop(shape, anchor, board)
        if anchor_new == anchor:
            return shape, anchor_new
        anchor = anchor_new


def rotate_left(shape, anchor, board):
    new_shape = rotated(shape, cclk=False)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def rotate_right(shape, anchor, board):
    new_shape = rotated(shape, cclk=True)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)

def idle(shape, anchor, board):
    return (shape, anchor)


class TetrisEnv(gym.Env):
    def __init__(self, width, height):
        super(TetrisEnv, self).__init__()
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=np.uint8)
        self.history_len = 4
        self.state = np.stack([self.board]*self.history_len)


        # actions are triggered by letters
        self.value_action_map = {
            0: left,
            1: right,
            2: hard_drop,
            3: soft_drop,
            4: rotate_left,
            5: rotate_right,
            6: idle,
        }
        self.action_value_map = dict([(j, i) for i, j in self.value_action_map.items()])
        self.nb_actions = len(self.value_action_map)

        # for running the engine
        self.time = -1
        self.score = -1
        self.anchor = None
        self.shape = None
        self.n_deaths = 0

        # used for generating shapes
        self._shape_counts = [0] * len(shapes)

        # clear after initializing
        self.reset()

        # Gym Data
        self.action_space = gym.spaces.Discrete(self.nb_actions)
        # self.observation_space = gym.spaces.MultiBinary(
        #     [self.width, self.height]
        # )
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(self.history_len, self.width, self.height), 
            dtype=np.uint8
        )
        self.reward_range = (-100, 100 * 4)

    def _choose_shape(self):
        maxm = max(self._shape_counts)
        m = [5 + maxm - x for x in self._shape_counts]
        r = random.randint(1, sum(m))
        for i, n in enumerate(m):
            r -= n
            if r <= 0:
                self._shape_counts[i] += 1
                return shapes[shape_names[i]]

    def _new_piece(self):
        # Place randomly on x-axis with 2 tiles padding
        #x = int((self.width/2+1) * np.random.rand(1,1)[0,0]) + 2
        self.anchor = (self.width / 2, 0)
        #self.anchor = (x, 0)
        self.shape = self._choose_shape()

    def _has_dropped(self):
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def _clear_lines(self):
        can_clear = [np.all(self.board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(self.board)
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                j -= 1
        self.score += sum(can_clear)
        self.board = new_board

        return sum(can_clear)

    def valid_action_count(self):
        valid_action_sum = 0

        for value, fn in self.value_action_map.items():
            # If they're equal, it is not a valid action
            if fn(self.shape, self.anchor, self.board) != (self.shape, self.anchor):
                valid_action_sum += 1

        return valid_action_sum

    def step(self, action: gym.spaces.Discrete):
        self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
        self.shape, self.anchor = self.value_action_map[action](self.shape, self.anchor, self.board)
        # Drop each step
        self.shape, self.anchor = soft_drop(self.shape, self.anchor, self.board)

        # Update time and reward
        self.time += 1
        reward = self.valid_action_count()
        #reward = random.randint(0, 0)
        reward = 1.

        done = False
        if self._has_dropped():
            self._set_piece(True)
            reward += 100. * self._clear_lines()
            if np.any(self.board[:, 0]):
                self.reset()
                self.n_deaths += 1
                done = True
                reward = -100.
            else:
                self._new_piece()

        self._set_piece(True)
        state = np.copy(self.board)
        self._set_piece(False)
        truncated = False
        info = {}
        self.state[1:,...] = self.state[:-1,...]
        self.state[0:,...] = state
        full_state = np.copy(self.state)
        return full_state, reward, done, info 

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> gym.spaces.MultiBinary:
        self.time = 0
        self.score = 0
        self._new_piece()
        self.board = np.zeros_like(self.board)
        state = np.copy(self.board)

        self.state = np.stack([self.board]*self.history_len)
        full_state = np.copy(self.state)
        # print(f"{full_state.shape = }")

        return full_state

    def close(self):
        pass

    def _set_piece(self, on=False):
        for i, j in self.shape:
            x, y = i + self.anchor[0], j + self.anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                self.board[int(self.anchor[0] + i), int(self.anchor[1] + j)] = on

    def render(self, mode='human'):
        def move_cursor(y, x):
            print(f"\033[{y};{x}H")
        move_cursor(1, 10 * 6)
        self._set_piece(True)
        s = 'o' + '-' * self.width + 'o\n'
        s += '\n'.join(['|' + ''.join(['X' if j else ' ' for j in i]) + '|' for i in self.board.T])
        s += '\no' + '-' * self.width + 'o'
        self._set_piece(False)
        print(s)
        