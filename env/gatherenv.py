from vizdoom import *
import numpy as np
import math
from gym.core import Env
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from model.util import *


class GatherEnv(Env):
    def __init__(self):
        self.game = DoomGame()
        self.game.load_config("O:\\Doom\\scenarios\\cig_flat.cfg")
        self.game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                           "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
                           "+viz_respawn_delay 1")
        self.game.add_game_args("+name AI +colorset 0")
        self.game.set_doom_map("map02")

        self.game.add_available_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        self.game.add_available_game_variable(GameVariable.HEALTH)
        self.game.add_available_game_variable(GameVariable.ARMOR)
        #self.game.set_labels_buffer_enabled(True)
        #self.game.set_depth_buffer_enabled(True)
        self.game.set_mode(Mode.PLAYER)
        self.game.init()
        self.action_space = Discrete(3)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)


    def reset(self):
        return get_observation(self.game.get_state(), real_frame=True)

    def seed(self, seed=None):
        self.game.set_seed(seed)
        return [seed]

    def step(self, action):
        one_hot_action = [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,0,0,1,0]]
        old_health = self.game.get_game_variable(GameVariable.HEALTH)
        old_ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        old_armor = self.game.get_game_variable(GameVariable.ARMOR)
        self.game.make_action(one_hot_action[action], 4)
        #for _ in range(4):
        #    self.game.advance_action()
        new_health = self.game.get_game_variable(GameVariable.HEALTH)
        new_ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        new_armor = self.game.get_game_variable(GameVariable.ARMOR)
        collection_rew = 0
        if new_health - old_health > 0:
            collection_rew += 1
        if new_ammo - old_ammo > 0:
            collection_rew += 1
        if new_armor - old_armor > 0:
            collection_rew += 1
        rew = collection_rew
        if self.game.is_player_dead():
            self.game.respawn_player()
        done = self.game.is_episode_finished()
        if done:
            self.game.new_episode()
        obs = get_observation(self.game.get_state(), real_frame=True)
        assert rew >= 0
        return obs, rew, done, None

    def render(self, mode='human'):
        pass

    def close(self):
        self.game.close()


if __name__ == '__main__':
    env = GatherEnv()
    rew = 0
    for i in range(1000):
        print(i)
        _, r, _, _ = env.step(0)
        rew += r
    print(rew / 1000)