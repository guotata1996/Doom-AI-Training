from vizdoom import *
import numpy as np
import math
from gym.core import Env
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import cv2
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2.policies import CnnPolicy
from baselines.a2c.a2c import Model
from model.util import *

class ShootEnv(Env):
    def __init__(self):
        self.game = DoomGame()
        self.game.load_config('O:\\Doom\\scenarios\\cig_flat2.cfg')
        self.game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                           "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
                           "+viz_respawn_delay 0")

        self.game.set_mode(Mode.PLAYER)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.add_available_game_variable(GameVariable.FRAGCOUNT)

        #define navigation env
        class NavigatorSubEnv(Env):
            def __init__(self, game):
                self.action_space = Discrete(3)
                self.observation_space = Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
                self._game = game

            def step(self, action):
                #-1 means it doesn't really controls the game
                if action > -1:
                    one_hot_action = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]]
                    self._game.make_action(one_hot_action[action], 4)
                    if self._game.is_episode_finished():
                        self._game.new_episode()
                    if self._game.is_player_dead():
                        self._game.respawn_player()

                obs = get_observation(self._game.get_state())
                return get_observation(self._game.get_state(), real_frame=True), 0, check_enemy_enter(obs), None

            def seed(self, seed=None):
                pass

            def reset(self):
                return get_observation(self._game.get_state(), real_frame=True)

            def render(self, mode='human'):
                pass

        self.navigator = VecFrameStack(VecEnvAdapter([NavigatorSubEnv(self.game)]), 4)

        #define navigation network
        self.navigation_policy = Model(CnnPolicy, self.navigator.observation_space, self.navigator.action_space, nenvs=1, nsteps=1)
        self.navigation_policy.load('O:\\Doom\\baselinemodel\\navigate_real2.dat')

        self.action_space = Discrete(3) #turn L, turn R, fire
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.available_actions = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,0,0,1]]


    def seed(self, seed=None):
        self.game.set_seed(seed)
        self.game.init()
        self.game.send_game_command("removebots")
        for i in range(8):
            self.game.send_game_command("addbot")

    def reset(self):
        obs_for_navigator = self.navigator.reset()
        while True:
            actions, _, _, _ = self.navigation_policy.step(obs_for_navigator)
            obs_for_navigator, _, navi_done, _ = self.navigator.step(actions)
            if navi_done:
                break
        obs = get_observation(self.game.get_state())
        assert check_enemy_enter(obs)
        return get_observation(self.game.get_state(), real_frame=True)

    def step(self, action):
        old_fragcount = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        self.game.make_action(self.available_actions[action], 4)
        new_fragcount = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        rew = new_fragcount - old_fragcount
        done = False

        if self.game.is_episode_finished():
            done = True
            self.game.new_episode()
            self.game.send_game_command("removebots")
            for i in range(8):
                self.game.send_game_command("addbot")

        if self.game.is_player_dead():
            self.game.respawn_player()
            done = True

        if action == 2:  # fire
            rew -= 0.05

        state = self.game.get_state()
        obs = get_observation(state)

        if check_enemy_enter(obs):
            rew += 0.01

        if check_enemy_leave(obs):
            done = True

        return get_observation(state, real_frame=True), rew, done, None
