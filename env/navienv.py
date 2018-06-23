from vizdoom import *
import numpy as np
import math
from gym.core import Env
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from model.util import *

x_bank = [-320, -120, 120]
y_bank = [-320, -120, 120]

class NaviEnv(Env):
    def __init__(self):
        self.game = DoomGame()
        self.game.load_config("O:\\Doom\\scenarios\\cig_flat.cfg")
        self.game.set_doom_scenario_path("O:\\Doom\\scenarios\\cig_flat_small.wad")
        self.game.set_doom_map("map01")
        self.game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                           "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
                           "+viz_respawn_delay 1")
        self.game.add_game_args("+name AI +colorset 0")
        self.game.set_doom_map("map02")
        self.game.add_available_game_variable(GameVariable.POSITION_X)
        self.game.add_available_game_variable(GameVariable.POSITION_Y)
        self.game.add_available_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        self.game.add_available_game_variable(GameVariable.HEALTH)
        self.game.add_available_game_variable(GameVariable.ARMOR)
        #self.game.set_labels_buffer_enabled(True)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_mode(Mode.PLAYER)
        self.game.init()
        self.action_space = Discrete(3)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

        self._reset_path_history()

    def reset(self):
        self.game.new_episode()
        self._reset_path_history()
        return get_observation(self.game.get_state(), real_frame=True)

    def seed(self, seed=None):
        self.game.set_seed(seed)
        return [seed]

    def step(self, action):
        one_hot_action = [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,0,0,1,0]]
        old_health = self.game.get_game_variable(GameVariable.HEALTH)
        old_ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        old_armor = self.game.get_game_variable(GameVariable.ARMOR)
        self.game.make_action(one_hot_action[action])
        new_health = self.game.get_game_variable(GameVariable.HEALTH)
        new_ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        new_armor = self.game.get_game_variable(GameVariable.ARMOR)
        location_rew = self._register_visit()
        collection_rew = 0
        if new_health - old_health > 0:
            collection_rew += 1
        if new_ammo - old_ammo > 0:
            collection_rew += 1
        if new_armor - old_armor > 0:
            collection_rew += 1

        depth_rew = 0
        if self.game.get_state() is not None:
            depth_buffer = self.game.get_state().depth_buffer.astype(np.int32)[:202, :]
            depth_sum = np.sum(np.max(depth_buffer, axis=0))
            if depth_sum < 10000:
                depth_rew = (depth_sum - 8000) / 80000
        rew = location_rew + collection_rew * 100 + depth_rew * 0.1
        done = self.game.is_episode_finished()
        if done:
            self.game.new_episode()
        obs = get_observation(self.game.get_state(), real_frame=True)
        return obs, rew, done, None

    def render(self, mode='human'):
        pass

    def close(self):
        self.game.close()

    def _reset_path_history(self):
        self.path_history = np.ones([3,3], dtype=np.float32)

    def _register_visit(self):
        self.path_history = np.maximum(self.path_history * 0.98, 1)
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        x_block, y_block = 0, 0
        for i in range(len(x_bank)):
            if pos_x < x_bank[i]:
                x_block = i
                break
        for i in range(len(y_bank)):
            if pos_y < y_bank[i]:
                y_block = i
                break
        if self.path_history[x_block, y_block] < 20:
            self.path_history[x_block, y_block] += 1
        return 1 / float(self.path_history[x_block, y_block])

if __name__ == '__main__':
    env = NaviEnv()
    rew = 0
    for i in range(1000):
        print(i)
        _, r, _, _ = env.step(0)
        rew += r
    print(rew / 1000)