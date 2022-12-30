#from __future__ import print_function

import pygame
from collections import deque
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

colors = {
    " ":                        (0,     0,      0),     # Game background
    "%":                        (75,    75,     75),    # Next pieces background
    ".":                        (35,    35,     35),    # Future drop color
    "T":                        (255,   85,     85),
    "J":                        (100,   200,    115),
    "L":                        (120,   108,    245),
    "Z":                        (255,   140,    50),
    "S":                        (50,    120,    52),
    "I":                        (146,   202,    73),
    "O":                        (150,   161,    218),
}

ainsi_colors = {sum(v): k for k, v in colors.items()}

shape_names = ['T', 'J', 'L', 'Z', 'S', 'I', 'O']


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


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
        if x < 0 or x >= board.shape[0] or y >= board.shape[1]:
            return True
        if np.all(board[x, y] != colors[' ']):
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


def optimize_TetrisEnv(trial):
    return {
        'coef_reward_for_piece_drop_height': trial.suggest_float('coef_reward_for_piece_drop_height', 0., 1., log=False),
        'coef_reward_for_piece_drop_bumps': trial.suggest_float('coef_reward_for_piece_drop_bumps', 0., 1., log=False),
        'coef_reward_for_piece_drop_holes': trial.suggest_float('coef_reward_for_piece_drop_holes', 0., 1., log=False),

        'coef_reward_for_step_drop_height': trial.suggest_float('coef_reward_for_step_drop_height', 0., 1., log=False),
        'coef_reward_for_step_drop_bumps': trial.suggest_float('coef_reward_for_step_drop_bumps', 0., 1., log=False),
        'coef_reward_for_step_drop_holes': trial.suggest_float('coef_reward_for_step_drop_holes', 0., 1., log=False),

        'reward_for_game_over': -trial.suggest_int('reward_for_game_over', 1, 10_000, log=True),
        'reward_for_piece_drop': trial.suggest_int('reward_for_piece_drop', 1, 100, log=True),
    }


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(
        self, 
        render_mode:str="human", 

        coef_reward_for_piece_drop_height:float=1.,
        coef_reward_for_piece_drop_bumps:float=1.,
        coef_reward_for_piece_drop_holes:float=1.,

        coef_reward_for_step_drop_height:float=0.,
        coef_reward_for_step_drop_bumps:float=0.,
        coef_reward_for_step_drop_holes:float=0.,

        reward_for_game_over:int=-1000,
        reward_for_piece_drop:int=1,
    ):
        super(TetrisEnv, self).__init__()
        self.render_mode = render_mode
        self.window = None
        self.width = 10
        self.height = 20
        self.full_width = self.width + 5
        self.full_height = self.height
        self.window_size = (self.full_width * 20, self.full_height * 20)
        self.channels = 3
        self.max_future_pieces = 4
        self.board = np.zeros(
            shape=(self.width, self.height, self.channels), 
            dtype=np.uint8
        )
        self.shapes = deque([], maxlen=self.max_future_pieces)
        self.drop_rate = 3
        self.turns_without_drop = 0

        self.reward_for_piece_drop = reward_for_piece_drop
        self.coef_reward_for_piece_drop_height = coef_reward_for_piece_drop_height
        self.coef_reward_for_piece_drop_bumps = coef_reward_for_piece_drop_bumps
        self.coef_reward_for_piece_drop_holes = coef_reward_for_piece_drop_holes

        self.coef_reward_for_step_drop_height = coef_reward_for_step_drop_height
        self.coef_reward_for_step_drop_bumps = coef_reward_for_step_drop_bumps
        self.coef_reward_for_step_drop_holes = coef_reward_for_step_drop_holes

        # actions are triggered by letters
        self.value_action_map = {
            0: idle,
            1: left,
            2: right,
            3: hard_drop,
            4: soft_drop,
            5: rotate_left,
            6: rotate_right,
        }
        self.score_for_lines_cleared = {
            0: 0.,
            1: 40.,
            2: 100.,      
            3: 300.,       
            4: 1200.,      
        }
        self.score_for_actions = {
            0: 0,
            1: 0,
            2: 0,
            3: 10.,
            4: 1.,
            5: 0,
            6: 0,
        }
        self.reward_for_game_over = reward_for_game_over
        self.action_value_map = dict([(j, i) for i, j in self.value_action_map.items()])
        self.nb_actions = len(self.value_action_map)

        # for running the engine
        self.time = -1
        self.score = -1
        self.anchor = None
        self.shape = None
        self.n_deaths = 0
        self.last_reward_modeling = 0
        self.total_dropped_pieces = 0

        # used for generating shapes
        self._shape_counts = [0] * len(shapes)

        # clear after initializing
        self.reset()

        # Gym Data
        self.action_space = gym.spaces.Discrete(self.nb_actions)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self.full_width, self.full_height, self.channels),
            dtype=np.uint8
        )
        self.reward_range = (
            self.reward_for_game_over,
            max(self.score_for_lines_cleared.values()) + max(self.score_for_actions.values()) + self.reward_for_piece_drop
        )

    def _reward_modeling_aggregate_height(self):
        s = (self.board.sum(axis=2) != 0).astype(np.int8)
        highest_block_in_cols = first_nonzero(s, axis=1, invalid_val=self.height)
        aggregate_height = (self.height - highest_block_in_cols).sum()
        return aggregate_height


    def _reward_modeling_complete_lines(self):
        s = (self.board.sum(axis=2) != 0).astype(np.int8)
        complete_lines = sum([np.all(s[:, i]) for i in range(self.height)])
        return complete_lines

    def _reward_modeling_holes(self):
        def count_holes(arr, highest_block_in_cols):
            cols_holes = []
            for i, top in enumerate(highest_block_in_cols):
                if top == self.height:
                    nb_holes = 0
                else:
                    nb_holes = (arr[i,top:] == 0).astype(np.int8).sum()
                    # print(f"{i = }, {nb_holes = }")
                cols_holes.append(nb_holes)
            return sum(cols_holes)
        s = (self.board.sum(axis=2) != 0).astype(np.int8)
        highest_block_in_cols = first_nonzero(s, axis=1, invalid_val=self.height)
        holes = count_holes(s, highest_block_in_cols)
        return holes

    def _reward_modeling_bumpiness(self):
        s = (self.board.sum(axis=2) != 0).astype(np.int8)
        highest_block_in_cols = first_nonzero(s, axis=1, invalid_val=self.height)
        bumpiness = 0
        for a, b in zip(highest_block_in_cols[1:], highest_block_in_cols[:-1]):
            bumpiness += abs(a - b)
        return bumpiness

    def _reward_modeling_has_tower(self):
        s = (self.board.sum(axis=2) != 0).astype(np.int8)
        highest_block_in_cols = first_nonzero(s, axis=1, invalid_val=self.height)
        if max(highest_block_in_cols) - min(highest_block_in_cols) > 4:
            return True
        # nb_of_towers = 0
        # for a, b in zip(highest_block_in_cols[1:], highest_block_in_cols[:-1]):
        #     if a - b > 4:
        #         nb_of_towers += 1
        # return nb_of_towers

    def _reward_modeling_for_piece(self):
        holes_a = self._reward_modeling_holes()
        bumps_a = self._reward_modeling_bumpiness()
        height_a = self._reward_modeling_aggregate_height()
        self._set_piece(on=True)
        holes_b = self._reward_modeling_holes()
        bumps_b = self._reward_modeling_bumpiness()
        height_b = self._reward_modeling_aggregate_height()
        self._set_piece(on=False)

        holes = (holes_a - holes_b) * self.coef_reward_for_piece_drop_holes
        bumps = (bumps_a - bumps_b) * self.coef_reward_for_piece_drop_bumps
        height = (height_a - height_b) * self.coef_reward_for_piece_drop_height
        return -float(holes + bumps + height)


    def _reward_modeling(self):
        aggregate_height = self._reward_modeling_complete_lines()
        # print(f"{aggregate_height = }")

        # complete_lines = self._reward_modeling_complete_lines()
        # print(f"{complete_lines = }")

        holes = self._reward_modeling_holes()
        # print(f"{holes = }")

        bumpiness = self._reward_modeling_bumpiness()

        # a = -0.510066
        # b = 0.760666
        # c = -0.35663
        # d = -0.184483
        a = self.coef_reward_for_step_drop_height
        c = self.coef_reward_for_step_drop_holes
        d = self.coef_reward_for_step_drop_bumps
        # e = 1
        # f = -1.

        # time_to_drop_piece = self.height
        # min_dropped_pieces = (self.time + 1) / time_to_drop_piece
        # dropped_pieces_advantage = self.total_dropped_pieces / min_dropped_pieces

        reward =    0. \
                    + a * aggregate_height \
                    + c * holes \
                    + d * bumpiness \
                    # + b * complete_lines \
                    # + e * self.total_dropped_pieces \
                    # + f * making_a_tower \
                    # + self.time

        # print(f"{reward = }")
        return float(reward)


    def _choose_shape(self):
        maxm = max(self._shape_counts)
        m = [5 + maxm - x for x in self._shape_counts]
        r = random.randint(1, sum(m))
        for i, n in enumerate(m):
            r -= n
            if r <= 0:
                self._shape_counts[i] += 1
                return shape_names[i]

    def _new_piece(self):
        # Place in center of x-axis, and the top of y-axis
        self.shapes.append(self._choose_shape())
        self.shape = shapes[self.shapes[0]]
        self.anchor = (int(self.width / 2), 0)

    def _has_dropped(self):
        """Check if the piece would be touching another block next turn."""
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def _clear_lines(self):
        can_clear = [np.all(self.board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(self.board)
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                j -= 1
        score = self.score_for_lines_cleared[sum(can_clear)]
        self.score += score
        self.board = new_board

        return score

    def _get_state(self):
        self._set_piece(True)
        board = np.copy(self.board)
        self._set_piece(False)
        state = np.zeros(
            shape=(self.full_width, self.full_height, self.channels),
            dtype=np.uint8
        )
        state[:board.shape[0],:board.shape[1]] = board
        state[board.shape[0]:,:] = colors["%"]
        for pice_nb, piece_name in enumerate(self.shapes):
            shape = shapes[piece_name]
            color = colors[piece_name]
            anchor = board.shape[0] + 2, ((pice_nb) * 5) + 3
            for i, j in shape:
                x, y = i + anchor[0], j + anchor[1]
                state[int(anchor[0] + i), int(anchor[1] + j)] = color
        # state = np.transpose(state, (2, 0, 1))
        return state

    
    def step(self, action: gym.spaces.Discrete):
        done = False
        reward = 1.

        self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
        self.shape, self.anchor = self.value_action_map[action](self.shape, self.anchor, self.board)
        # reward += self.score_for_actions[action]
        
        self.turns_without_drop += 1 
        if action in [3, 4]:
            self.turns_without_drop = 0

        reward += self._reward_modeling()
        # reward_modeling = self._reward_modeling()
        # reward += reward_modeling - self.last_reward_modeling
        # self.last_reward_modeling = reward_modeling
        # reward -= self._reward_modeling_bumpiness()
        # has_tower = self._reward_modeling_has_tower()
        # if has_tower:
        #     done = True

        # Drop each step
        if self.turns_without_drop >= self.drop_rate:
            self.shape, self.anchor = soft_drop(self.shape, self.anchor, self.board)
            self.turns_without_drop = 0

        # Update time and reward
        self.time += 1
        if self._has_dropped():
            reward += self.reward_for_piece_drop
            reward += self._reward_modeling_for_piece()
            self.total_dropped_pieces += 1
            self._set_piece(True)
            reward += self._clear_lines()
            if np.any(self.board[:, 0]):
                done = True
            else:
                self._new_piece()

        if done == True:
            self.reset()
            self.n_deaths += 1
            reward += self.reward_for_game_over

        state = self._get_state()
        info = {
            "steps": self.time,
            "pieces": self.total_dropped_pieces,
            "score": self.score,
            "turns_without_drop": self.turns_without_drop,
        }
        return state, reward, done, info 


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> gym.spaces.MultiBinary:
        self.time = 0
        self.score = 0
        self.last_reward_modeling = 0
        self.total_dropped_pieces = 0
        self.turns_without_drop = 0
        for _ in range(self.max_future_pieces):
            self._new_piece()
        self.board = np.zeros_like(self.board)
        state = self._get_state()
        return state


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


    def _set_piece(self, on=False):
        if on == False:
            self.board[self.board == colors["."]] = 0
        shape, anchor = hard_drop(self.shape, self.anchor, self.board)
        for i, j in self.shape:
            x, y = i + self.anchor[0], j + self.anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                if on == True:
                    self.board[x, y] = colors[self.shapes[0]]
                if on == False:
                    self.board[x, y] = colors[" "]
        for i, j in shape:
            x, y = i + anchor[0], j + anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                if np.all(self.board[x, y] == colors[" "]):
                    if on == True:
                        self.board[x, y] = colors["."]
                
    
    def _render_frame(self, render_mode:str):
        if render_mode == "human":
            if self.window is None and render_mode == "human":
                pygame.init()
                pygame.display.init()
                self.clock = None
                self.window = pygame.display.set_mode(size=self.window_size)
            if self.clock is None and render_mode == "human":
                self.clock = pygame.time.Clock()
                self.canvas = pygame.Surface(self.window_size)
        
            # self._set_piece(True)
            state = self._get_state()
            # self._set_piece(False)
            surf = pygame.surfarray.make_surface(state)
            picture = pygame.transform.scale(surf, self.window_size)
            self.window.blit(picture, (0, 0))
            pygame.display.update()
            self.clock.tick(120)
            return state
        elif render_mode == "rgb_array":
            state = self._get_state()
            return state


    def render(self, mode=None):
        if mode == None:
            mode = self.render_mode
        if mode in ["human", "rgb_array"]:
            frame = self._render_frame(render_mode=mode)
            return frame.T
        elif mode == "ansi":
            def move_cursor(y, x):
                print(f"\033[{y};{x}H")
            self._set_piece(True)
            s = 'o' + '-' * self.width + 'o\n'
            s += '\n'.join(['|' + ''.join(['X' if j else ' ' for j in i]) + '|' for i in self.board.T])
            s += '\no' + '-' * self.width + 'o'
            self._set_piece(False)
            move_cursor(1, 10 * 6)
            print(s)
            return s
