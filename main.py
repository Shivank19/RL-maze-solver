import itertools
import time
import numpy as np
import cv2
from moviepy.editor import VideoClip

GRID_HEIGHT = 16
GRID_WIDTH = 9
WALL_FRAC = .2
NUM_WINS = 5
NUM_LOSE = 10

class Grid:
    def __init__(self, grid_height=3, grid_width=4, discount_factor=.5, default_reward=-.5, wall_penalty=-.6, 
                 win_reward=5., lose_reward=-10., viz=True, patch_side=120, grid_thickness=2, arrow_thickness=3,
                 wall_locs=[[1, 1], [1, 2]], win_locs=[[0, 3]], lose_locs=[[1, 3]], start_loc=[0, 0],
                 reset_prob=.2):
                self.grid = np.ones([grid_height, grid_width]) * default_reward
                self.reset_prob = reset_prob
                self.grid_height = grid_height
                self.grid_width = grid_width
                self.wall_penalty = wall_penalty
                self.win_reward = win_reward
                self.lose_reward = lose_reward
                self.default_reward = default_reward
                self.discount_factor = discount_factor
                self.patch_side = patch_side
                self.grid_thickness = grid_thickness
                self.arrow_thickness = arrow_thickness
                self.wall_locs = np.array(wall_locs)
                self.win_locs = np.array(win_locs)
                self.lose_locs = np.array(lose_locs)
                self.at_terminal_state = False
                self.auto_reset = True
                self.random_respawn = True
                self.step = 0
                self.viz_canvas = None
                self.viz = viz
                self.path_color = (128, 128, 128)
                self.wall_color = (0, 255, 0)
                self.win_color = (0, 0, 255)
                self.lose_color = (255, 0, 0)
                self.grid[self.wall_locs[:, 0], self.wall_locs[:, 1]] = self.wall_penalty
                self.grid[self.lose_locs[:, 0], self.lose_locs[:, 1]] = self.lose_reward
                self.grid[self.win_locs[:, 0], self.win_locs[:, 1]] = self.win_reward
                spawn_condn = lambda loc: self.grid[loc[0], loc[1]] == self.default_reward
                self.spawn_locs = np.array([loc for loc in itertools.product(np.arange(self.grid_height), np.arange(self.grid_width)) if spawn_condn(loc) ])

                self.start_state = np.array(start_loc)
                self.bot_rc = None
                self.reset()
                self.actions = [self.up, self.left, self.right, self.down, self.noop]
                self.action_labels = ['UP', 'LEFT', 'RIGHT', 'DOWN', 'NOOP']

                self.q_values = np.ones([self.grid.shape[0], self.grid.shape[1], len(self.actions)]) * 1 / len(self.actions)

                if self.viz:
                    self.init_grid_canvas()
                    self.video_out_fpath = 'RL_DQN_GridSolver-' + str(time.time()) + '.mp4'
                    self.clip = VideoClip(self.make_frame, duration=15)                    

    def make_frame(self, t):
        self.action()
        frame = self.highlight_loc(self.viz_canvas, self.bot_rc[0], self.bot_rc[1])
        return frame
    
    def check_terminal_state(self):
        if self.grid[self.bot_rc[0], self.bot_rc[1]] == self.lose_reward or self.grid[self.bot_rc[0], self.bot_rc[1]] == self.win_reward:
            self.at_terminal_state = True
            if self.auto_reset:
                self.reset()
    
    def reset(self):
        if not self.random_respawn:
            self.bot_rc = self.start_state.copy()
        else:
            self.bot_rc = self.spawn_locs[np.random.choice(np.arange(len(self.spawn_locs)))].copy()
        self.at_terminal_state = False
    
    def up(self):
        action_idx = 0
        new_r = self.bot_rc[0] - 1
        if new_r < 0 or self.grid[new_r, self.bot_rc[1]] == self.wall_penalty:
            return self.wall_penalty, action_idx
        self.bot_rc[0] = new_r
        reward = self.grid[self.bot_rc[0], self.bot_rc[1]]
        self.check_terminal_state()
        return reward, action_idx
    
    def left(self):
        action_idx = 1
        new_c = self.bot_rc[1] - 1
        if new_c < 0 or self.grid[self.bot_rc[0], new_c] == self.wall_penalty:
            return self.wall_penalty, action_idx
        self.bot_rc[1] = new_c
        reward = self.grid[self.bot_rc[0], self.bot_rc[1]]
        self.check_terminal_state()
        return reward, action_idx
    
    def right(self):
        action_idx = 2
        new_c = self.bot_rc[1] + 1
        if new_c >= self.grid.shape[1] or self.grid[self.bot_rc[0], new_c] == self.wall_penalty:
            return self.wall_penalty, action_idx
        self.bot_rc[1] = new_c
        reward = self.grid[self.bot_rc[0], self.bot_rc[1]]
        self.check_terminal_state()
        return reward, action_idx

    def down(self):
        action_idx = 3
        new_r = self.bot_rc[0] + 1
        if new_r >= self.grid.shape[0] or self.grid[new_r, self.bot_rc[1]] == self.wall_penalty:
            return self.wall_penalty, action_idx
        self.bot_rc[0] = new_r
        reward = self.grid[self.bot_rc[0], self.bot_rc[1]]
        self.check_terminal_state()
        return reward, action_idx

    def noop(self):
        action_idx = 4
        reward = self.grid[self.bot_rc[0], self.bot_rc[1]]
        self.check_terminal_state()
        return reward, action_idx

    def q_vals_to_probs(self, q_vals, epsilon=1e-4):
        """Converts Q values to probabilities"""
        action_probs = q_vals - q_vals.min() + epsilon
        action_probs = action_probs / action_probs.sum()
        return action_probs
    
    def action(self):
        print('================ ACTION =================')
        if self.at_terminal_state:
            print('At terminal state, please call reset()')
            exit()
        start_bot_rc = self.bot_rc[0], self.bot_rc[1]
        q_vals = self.q_values[self.bot_rc[0], self.bot_rc[1]]
        action_probs = self.q_vals_to_probs(q_vals)
        reward, action_idx = np.random.choice(self.actions, p=action_probs)()
        print('End position:', self.bot_rc)
        print('Reward:', reward)
        alpha = np.exp(-self.step / 10e9)
        self.step += 1
        qv = (1 - alpha) * q_vals[action_idx] + alpha * (reward + self.discount_factor * self.q_values[self.bot_rc[0], self.bot_rc[1]].max())
        self.q_values[start_bot_rc[0], start_bot_rc[1], action_idx] = qv
        if self.viz:
            self.update_viz(start_bot_rc[0], start_bot_rc[1])
        if np.random.rand() < self.reset_prob:
            print('-----> Randomly resetting to a random spawn point with probability', self.reset_prob)
            self.reset()

    def highlight_loc(self, viz_in, i, j):
        start_y = i * (self.patch_side + self.grid_thickness)
        end_y = start_y + self.patch_side
        start_x = j * (self.patch_side + self.grid_thickness)
        end_x = start_x + self.patch_side
        viz = viz_in.copy()
        cv2.rectangle(viz, (start_x, start_y), (end_x, end_y), (255, 255, 255), thickness=self.grid_thickness)
        return viz

    def update_viz(self, i, j):
        start_y = i * (self.patch_side + self.grid_thickness)
        end_y = start_y + self.patch_side
        start_x = j * (self.patch_side + self.grid_thickness)
        end_x = start_x + self.patch_side
        patch = np.zeros([self.patch_side, self.patch_side, 3]).astype(np.uint8)

        if self.grid[i, j] == self.default_reward:
            patch[:, :, :] = self.path_color
        elif self.grid[i, j] == self.wall_penalty:
            patch[:, :, :] = self.wall_color
        elif self.grid[i, j] == self.win_reward:
            patch[:, :, :] = self.win_color
        elif self.grid[i, j] == self.lose_reward:
            patch[:, :, :] = self.lose_color
        
        if self.grid[i,  j] == self.default_reward:
            action_probs = self.q_vals_to_probs(self.q_values[i, j])
            x_comp = action_probs[2] - action_probs[1]
            y_comp = action_probs[0] - action_probs[3]
            magnitude = 1. - action_probs[-1]
            s = self.patch_side // 2
            x_patch = int(s * x_comp)
            y_patch = int(s * y_comp)
            arrow_canvas = np.zeros_like(patch)
            vx = s + x_patch
            vy = s - x_patch
            cv2.arrowedLine(arrow_canvas, (s, s), (vx, vy), (255, 255, 255), thickness=self.arrow_thickness, tipLength=0.5)
            grid_box = (magnitude * arrow_canvas + (1 - magnitude) * patch).astype(np.uint8)
            self.viz_canvas[start_y : end_y, start_x : end_x] = grid_box
        else:
            self.viz_canvas[start_y:end_y, start_x:end_x] = patch

    def init_grid_canvas(self):
        org_h, org_w = self.grid_height, self.grid_width
        viz_w = (self.patch_side * org_w) + (self.grid_thickness * (org_w - 1))
        viz_h = (self.patch_side * org_h) + (self.grid_thickness * (org_h - 1))
        self.viz_canvas = np.zeros([viz_h, viz_w, 3]).astype(np.uint8)
        for i in range(org_h):
            for j in range(org_w):
                self.update_viz(i, j)

    def solve(self):
        if not self.viz:
            while True:
                self.action()
        else:
            self.clip.write_videofile(self.video_out_fpath, fps = 460)

def gen_grid_config(h, w, wall_frac=.5, num_wins=2, num_lose=3):
    n = h * w
    num_wall_blocks = int(wall_frac * n)
    wall_locs = (np.random.rand(num_wall_blocks, 2) * [h, w]).astype(np.int)
    win_locs = (np.random.rand(num_wins, 2) * [h, w]).astype(np.int)
    lose_locs = (np.random.rand(num_lose, 2) * [h, w]).astype(np.int)
    return wall_locs, win_locs, lose_locs

if __name__ == '__main__':
    wall_locs, win_locs, lose_locs = gen_grid_config(GRID_HEIGHT, GRID_WIDTH, WALL_FRAC, NUM_WINS, NUM_LOSE)
    g = Grid(grid_height = GRID_HEIGHT, grid_width = GRID_WIDTH, wall_locs = wall_locs, win_locs = win_locs, lose_locs = lose_locs, viz = True)
    g.solve()
    k = 0
