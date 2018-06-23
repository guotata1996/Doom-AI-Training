from vizdoom import *
from gym.core import Env
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from model.util import *
import numpy as np
import time

class DodgeEnv(Env):
    def __init__(self):
        self.game = DoomGame()
        self.game.load_config("O:\\Doom\\a2c\\scenarios\\dodge\\dodge.cfg")
        self.game.set_doom_scenario_path("O:\\Doom\\a2c\\scenarios\\dodge\\dodge1.wad")
        #self.game.load_config('O:\\Doom\\scenarios\\cig_flat.cfg')
        #self.game.set_doom_scenario_path('O:\\Doom\\scenarios\\cig_flat_small.wad')
        #self.game.set_doom_map("map03")

        self.game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                                "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
                                "+viz_respawn_delay 0")

        self.game.set_mode(Mode.PLAYER)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)

        self.action_space = Discrete(3)
        self.observation_space = Box(low=0, high=255, shape=(168, 168, 3), dtype=np.uint8)
        self.available_actions = [[1,0],[0,1],[0,0]]
        #self.available_actions = [[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0]]
        self.bots = 1

    def seed(self, seed=None):
        self.game.set_seed(seed)
        self.game.init()
        self.game.send_game_command("removebots")
        for i in range(self.bots):
            self.game.send_game_command("addbot SlowShoot{}".format(i))

    def reset(self):
        return get_observation(self.game.get_state(), real_frame=True, resolution=168)

    def step(self, action):
        rew = 0
        if action == 2:
            rew += 0.05 #living reward
        old_health = self.game.get_game_variable(GameVariable.HEALTH)
        self.game.make_action(self.available_actions[action], 4)
        new_health = self.game.get_game_variable(GameVariable.HEALTH)
        rew -= -0.01*(min(new_health - old_health, 0)) #health reward <= 0
        done = False

        if self.game.is_episode_finished():
            done = True
            self.game.new_episode()
            self.game.send_game_command("removebots")
            for i in range(self.bots):
                self.game.send_game_command("addbot SlowShoot{}".format(i))

        if self.game.is_player_dead():
            self.game.respawn_player()
            done = True

        state = self.game.get_state()
        return get_observation(state, real_frame=True, resolution=168), rew, done, None