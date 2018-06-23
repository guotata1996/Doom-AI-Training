from baselines.a2c.utils import find_trainable_variables
from baselines.common import tf_util
from baselines.ppo2.policies import CnnPolicy, LstmPolicy
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import numpy as np
import joblib
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from env.mixedenv import MixedEnv
import cv2
from model.util import check_enemy_enter, check_bullet
import time

def make_subproc_env():
    def make_env():
        def _thunk():
            env = MixedEnv()
            env.seed(100)
            return env
        return _thunk
    return SubprocVecEnv([make_env()])

if __name__ == '__main__':
    sess = tf_util.make_session()

    restores = []
    navigation_model = LstmPolicy(sess, Box(low=0, high=255, shape=(84, 84, 12), dtype=np.uint8),
                                  Discrete(3), 1, 1, reuse=False, model_name='navi')
    navigation_params = find_trainable_variables('navi')
    navigation_loaded = joblib.load('O:\\Doom\\a2c\\scenarios\\display\\navi.dat')
    for p, loaded_p in zip(navigation_params, navigation_loaded):
        restores.append(p.assign(loaded_p))

    shoot_model = LstmPolicy(sess, Box(low=0, high=255, shape=(168, 168, 12), dtype=np.uint8),
                             Discrete(3), 1, 1, reuse=False, model_name='shoot')
    shoot_params = find_trainable_variables('shoot')
    shoot_loaded = joblib.load('O:\\Doom\\a2c\\scenarios\\display\\shoot.dat')
    for p, loaded_p in zip(shoot_params, shoot_loaded):
        restores.append(p.assign(loaded_p))

    dodge_model = LstmPolicy(sess, Box(low=0, high=255, shape=(168, 168, 12), dtype=np.uint8),
                             Discrete(3), 1, 1, reuse=False, model_name='dodge')
    dodge_params = find_trainable_variables('dodge')
    dodge_loaded = joblib.load('O:\\Doom\\a2c\\scenarios\\display\\dodge.dat')
    for p, loaded_p in zip(dodge_params, dodge_loaded):
        restores.append(p.assign(loaded_p))

    ps = sess.run(restores)

    navigation_lstmstate = navigation_model.initial_state
    shoot_lstmstate = shoot_model.initial_state
    dodge_lstmstate = dodge_model.initial_state

    env = VecFrameStack(make_subproc_env(), 4)

    obs = env.reset()

    controller = 'navi'

    total_reward = 0
    last_action = None
    dodge_countdown = 0

    for step in range(10000):
        frame = obs[0,:,:168,:]
        label = obs[0,:,168:,0]

        if controller == 'navi':
            if check_enemy_enter(label):
                controller = 'shoot'

                shoot_lstmstate = shoot_model.initial_state
            elif check_bullet(label):
                controller = 'dodge'
                dodge_lstmstate = dodge_model.initial_state
        elif controller == 'shoot':
            if not check_enemy_enter(label):
                navigation_lstmstate = navigation_model.initial_state
                controller = 'navi'
            if last_action == 5:
                controller = 'dodge'
                dodge_lstmstate = dodge_model.initial_state
                dodge_countdown = 8
        else:
            dodge_countdown -= 1
            if dodge_countdown == 0:
                if check_enemy_enter(label):
                    controller = 'shoot'
                    shoot_lstmstate = shoot_model.initial_state
                else:
                    controller = 'navi'
                    navigation_lstmstate = navigation_model.initial_state

        if controller == 'shoot':
            #print('shoot model running')
            ac, _, shoot_lstmstate, _ = shoot_model.step([frame], shoot_lstmstate, [False])
            ac = ac[0]
            cvt_table = [0,1,5]
            real_ac = cvt_table[ac]
            last_action = real_ac
        elif controller == 'navi':
            #print('navigation model running')
            small_frame = cv2.resize(frame, (84, 84))
            ac, _, navigation_lstmstate, _ = navigation_model.step([small_frame], navigation_lstmstate, [False])
            ac = ac[0]
            cvt_table = [0,1,4]
            real_ac = cvt_table[ac]
            last_action = real_ac
        else:
            #print('dodge model running')
            ac, _, navigation_lstmstate, _ = dodge_model.step([frame], dodge_lstmstate, [False])
            ac = ac[0]
            cvt_table = [3,2,6]
            real_ac = cvt_table[ac]
            last_action = real_ac

        obs, rewards, _, _ = env.step([real_ac])
        total_reward += rewards[0]
        #time.sleep(0.1)

    print(total_reward / 2000)