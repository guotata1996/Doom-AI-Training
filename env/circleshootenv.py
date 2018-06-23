from vizdoom import *
from gym.core import Env
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from model.util import *
import numpy as np

class ShootEnv(Env):
    def __init__(self):
        self.game = DoomGame()
        self.game.load_config("O:\\Doom\\a2c\\scenarios\\shoot_circle.cfg")
        self.game.set_doom_scenario_path("O:\\Doom\\a2c\\scenarios\\shoot_circle4.wad")
        #self.game.load_config('O:\\Doom\\scenarios\\cig_flat.cfg')
        #self.game.set_doom_scenario_path('O:\\Doom\\scenarios\\cig_flat_small.wad')
        self.game.set_doom_map("map01")

        self.game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                                "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
                                "+viz_respawn_delay 0")
        self.action_space = Discrete(3)
        self.observation_space = Box(low=0, high=255, shape=(168, 168, 3), dtype=np.uint8)
        #self.available_actions = np.eye(3).tolist()
        #self.bots_count = 3
        self.available_actions = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]]
        self.bots_count = 3

    def seed(self, seed=None):
        self.game.set_seed(seed)
        self.game.init()
        self.game.send_game_command("removebots")
        for i in range(self.bots_count):
            self.game.send_game_command("addbot")

    def reset(self):
        return get_observation(self.game.get_state(), real_frame=True, resolution=168)

    def step(self, action):
        old_frag = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        self.game.make_action(self.available_actions[action])
        rew = self.game.get_game_variable(GameVariable.FRAGCOUNT) - old_frag

        done = False
        if self.game.is_player_dead():
            self.game.respawn_player()
            done = True
        if self.game.is_episode_finished():
            done = True
            self.game.new_episode()
            self.game.send_game_command("removebots")
            for i in range(self.bots_count):
                self.game.send_game_command("addbot")
        if action == 2:
            rew -= 0.1

        return get_observation(self.game.get_state(), real_frame=True, resolution=168), rew, done, None