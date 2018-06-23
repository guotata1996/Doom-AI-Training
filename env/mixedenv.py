from vizdoom import *
from gym.core import Env
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from model.util import *

class MixedEnv(Env):
    def __init__(self):
        self.game = DoomGame()
        self.game.load_config('O:\\Doom\\scenarios\\cig.cfg')
        self.game.set_doom_scenario_path('O:\\Doom\\scenarios\\cig_flat_small.wad')
        self.game.set_seed(0)
        self.game.set_doom_map("map01")
        #self.game.set_doom_map("map02")
        self.game.add_game_args("-host 1 -deathmatch +timelimit 10.0 "
                           "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
                           "+viz_respawn_delay 1")

        self.game.set_mode(Mode.ASYNC_PLAYER)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.add_available_game_variable(GameVariable.FRAGCOUNT)
        self.game.add_available_game_variable(GameVariable.DEATHCOUNT)

        self.action_space = Discrete(6)
        self.observation_space = Box(low=0, high=255, shape=(168, 168*2, 3), dtype=np.uint8)
        #self.observation_space = Box(low=0, high=255, shape=(168, 168, 3), dtype=np.uint8)
        self.available_actions = np.eye(6).tolist()
        self.available_actions.append([0,0,0,0,0,0])
        self.bots_count = 4
        self.step_cnt = 0

    def seed(self, seed=None):
        self.game.set_seed(seed)
        self.game.init()
        self.game.send_game_command("removebots")
        for i in range(self.bots_count):
            self.game.send_game_command("addbot")

    def reset(self):
        #168 frame, 84 frame, enemy_in_view
        obs1 = get_observation(self.game.get_state(), real_frame=True, resolution=168)
        obs2 = get_observation(self.game.get_state(), real_frame=False, resolution=168)
        obs2 = np.concatenate([obs2, obs2, obs2], axis=2)
        assert obs2.shape == (168, 168, 3)
        obs = np.concatenate([obs1, obs2], axis=1)
        assert obs.shape == (168, 168*2, 3)
        return obs

    def step(self, action):
        self.step_cnt += 1
        if self.step_cnt % 100 == 0:
            print(self.step_cnt)
            print(self.game.get_game_variable(GameVariable.FRAGCOUNT), self.game.get_game_variable(GameVariable.DEATHCOUNT))

        old_frag = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        self.game.make_action(self.available_actions[action])
        new_frag = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        rew = new_frag - old_frag
        if action == 5:
            rew -= 0.1
        done = False

        if self.game.is_episode_finished():
            print("epi finished")
            done = True
            self.game.new_episode()
            self.game.send_game_command("removebots")
            for i in range(self.bots_count):
                self.game.send_game_command("addbot")

        if self.game.is_player_dead():
            self.game.respawn_player()
            done = True

        obs1 = get_observation(self.game.get_state(), real_frame=True, resolution=168)
        obs2 = get_observation(self.game.get_state(), real_frame=False, resolution=168)
        obs2 = np.concatenate([obs2, obs2, obs2], axis=2)
        assert obs2.shape == (168, 168, 3)
        obs = np.concatenate([obs1, obs2], axis=1)
        assert obs.shape == (168, 168*2, 3)

        return obs, rew, done, None

