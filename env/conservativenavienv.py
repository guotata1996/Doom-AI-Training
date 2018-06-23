from vizdoom import *
import numpy as np
import math
from gym.core import Env
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from model.util import *
import cv2
import joblib

x_bank = [-352, -180, 0, 180, 352, 481]
y_bank = [-320, -180, 0, 180, 320, 481]


def _get_depth_info(game):
    # only call this func after init
    pos_x = game.get_game_variable(GameVariable.POSITION_X)
    pos_y = game.get_game_variable(GameVariable.POSITION_Y)
    angle = game.get_game_variable(GameVariable.ANGLE)
    depth_buffer = game.get_state().depth_buffer.astype(np.int32)[:202, :]
    depth = np.max(depth_buffer, axis=0)

    if angle > (345 + 360) / 2:
        ag_idx = 0
    else:
        ag_idx = round(angle / 15)
    return min((round(pos_x) + 480) // 32, 30), min((round(pos_y) + 480) // 32, 30), ag_idx, np.sum(depth)

class ConservativeNaviEnv(Env):
    def __init__(self):
        self.game = DoomGame()
        self.game.load_config("O:\\Doom\\scenarios\\cig_flat2.cfg")
        self.game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                           "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
                           "+viz_respawn_delay 1")
        self.game.add_game_args("+name AI +colorset 0")

        self.game.add_available_game_variable(GameVariable.POSITION_X)
        self.game.add_available_game_variable(GameVariable.POSITION_Y)

        self.game.set_depth_buffer_enabled(True)
        #self.game.set_mode(Mode.SPECTATOR)
        self.game.init()
        self.action_space = Discrete(5)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

        self._reset_path_history()
        self.depthinfo = joblib.load('O:\\Doom\\mapinfo\\visdepth.dat')

    def reset(self):
        self.game.new_episode()
        self._reset_path_history()
        return get_observation(self.game.get_state(), real_frame=True)

    def seed(self, seed=None):
        self.game.set_seed(seed)
        return [seed]

    def step(self, action):
        one_hot_action = [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0]]

        self.game.make_action(one_hot_action[action], 4)
        #for _ in range(4):
        #    self.game.advance_action()
        done = self.game.is_episode_finished()
        if done:
            self.game.new_episode()

        location_rew = self._register_visit()
        x, y, angle, depth = _get_depth_info(self.game)
        counter_depth = [self.depthinfo[x, y, (angle + 6 * k) % 24] for k in range(1, 4)]
        diff = depth - np.mean(counter_depth)
        dep_rew = np.clip(diff / 22000, -1, 1)

        rew = location_rew + dep_rew

        obs = get_observation(self.game.get_state(), real_frame=True)
        return obs, rew, done, None

    def render(self, mode='human'):
        pass

    def close(self):
        self.game.close()

    def _reset_path_history(self):
        self.path_history = np.ones([30,30], dtype=np.float32)

    def _register_visit(self):
        self.path_history = np.maximum(self.path_history * 0.95, 1)
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        '''
        x_block, y_block = 0, 0
        for i in range(len(x_bank)):
            if pos_x < x_bank[i]:
                x_block = i
                break
        for i in range(len(y_bank)):
            if pos_y < y_bank[i]:
                y_block = i
                break
        '''
        x_block = min(round(pos_x + 480) // 32, 29)
        y_block = min(round(pos_y + 480) // 32, 29)
        if self.path_history[x_block, y_block] < 3:
            self.path_history[x_block, y_block] += 1
        return 1 / float(self.path_history[x_block, y_block])


if __name__ == '__main__':
    if False:
        game = DoomGame()
        game.load_config("O:\\Doom\\scenarios\\cig_flat2.cfg")

        game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                            "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
                            "+viz_respawn_delay 1")
        game.add_game_args("+name AI +colorset 0")
        game.set_mode(Mode.SPECTATOR)
        game.set_depth_buffer_enabled(True)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.add_available_game_variable(GameVariable.ANGLE)
        game.init()

        episodes = 10

        #should vary according to X_len,Y_len
        #recording = np.zeros([960//32,  960//32, 360 // 15], dtype=np.uint16)
        recording = joblib.load('O:\\Doom\\mapinfo\\visdepth.dat')
        step = 0
        for _ in range(episodes):
            while not game.is_episode_finished():
                x_n, y_n, _, _ = _get_depth_info(game)
                '''
                if np.sum(recording[x_n, y_n, :] > 0) < 24:
                    game.make_action([1,0,0,0,0,0])
                else:
                    av_actions = [[0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0]]
                    ac = np.random.choice(range(3))
                    game.make_action(av_actions[ac])
                '''
                game.advance_action()
                if game.is_episode_finished():
                    game.new_episode()
                x_, y_, ag_, dep_ = _get_depth_info(game)
                recording[x_, y_, ag_] = dep_
                cnt = np.sum(recording > 0, axis=-1)
                cnt = cnt.astype(np.uint8) * 10
                cnt[x_n, y_n] = 255
                cnt = np.repeat(cnt, 10, axis=0)
                cnt = np.repeat(cnt, 10, axis=1)
                cv2.imshow('completed', cnt)
                cv2.waitKey(1)

                step += 1
                if step == 100:
                    step = 0
                    joblib.dump(recording, 'O:\\Doom\\mapinfo\\visdepth.dat')
            game.new_episode()

    if True:
        env = ConservativeNaviEnv()
        rew = 0
        for i in range(1000):
            print(i)
            _, r, _, _ = env.step(0)
            rew += r
        print(rew / 1000)

