from baselines.a2c.a2c import Runner, Model
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from run_doom import make_doom_env
from model.envNetwork import EnvNetwork
from baselines.ppo2.policies import CnnPolicy
import numpy as np
from tqdm import tqdm
import cv2
from gym.spaces.discrete import Discrete
from model.mixedmodel import MixedModel
from model.util import check_enemy_leave, check_enemy_enter
from baselines.common import tf_util
from model.util import norm_factor
import tensorflow as tf

K = 5
num_env = 16
nsteps = 92
save_freq = 200
singlestep = 8
total_timesteps = int(80e6)
seed = 0

valid_batch_size = num_env * singlestep * (nsteps // singlestep - 1)

def main(visualize=False):
    session = tf_util.make_session()
    env_model = EnvNetwork(action_space_size=6, nbatch=num_env*singlestep, K=K, nsteps=singlestep, reuse=False, session=session)
    session.run(tf.global_variables_initializer())
    env_model.restore()

    env = VecFrameStack(make_doom_env(num_env, seed,'mixed'), 4)
    navi_model = Model(policy=CnnPolicy, ob_space=env.observation_space, ac_space=Discrete(3),
                         nenvs=num_env, nsteps=nsteps, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                         lr = 7e-4, alpha=0.99, epsilon=1e-5, total_timesteps=total_timesteps, lrschedule='linear',  model_name='navi')
    navi_model.load("O:\\Doom\\baselinemodel\\navigate_flat2.dat")

    fire_model = Model(policy=CnnPolicy, ob_space=env.observation_space, ac_space=Discrete(3),
                         nenvs=num_env, nsteps=nsteps, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                         lr = 7e-4, alpha=0.99, epsilon=1e-5, total_timesteps=total_timesteps, lrschedule='linear',  model_name='fire')

    fire_model.load("O:\\Doom\\baselinemodel\\fire_flat2.dat")
    policy_model = MixedModel(navi_model, fire_model, check_enemy_leave, check_enemy_enter, [0, 1, 4], [0, 1, 5])
    runner = Runner(env, policy_model, nsteps=nsteps, gamma=0.99)

    nh, nw, nc = env.observation_space.shape

    while True:
        total_loss = 0
        for _ in tqdm(range(save_freq)):
            obs1, _, _, mask1 , actions1, _ = runner.run()

            obs1 = np.reshape(obs1, [num_env, nsteps, nh, nw, nc])
            obs1 = obs1[:,:,:,:,-1:]

            actions1 = np.reshape(actions1, [num_env, nsteps])
            mask1 = np.reshape(mask1, [num_env, nsteps])

            hidden_states = env_model.initial_state
            for s in range(0, nsteps - K - singlestep, singlestep):
                input_frames = obs1[:,s:s+singlestep,:,:,:] //norm_factor
                input_frames = np.reshape(input_frames, [num_env*singlestep, nh, nw])
                input_frames = np.eye(9)[input_frames]
                actions, masks, expected_observations = [], [], []
                for t in range(K):
                    expected_observation = obs1[:,s+t+1 : s+singlestep+t+1, :,:,:]
                    expected_observation = np.reshape(expected_observation, [num_env*singlestep, nh, nw, 1])
                    expected_observations.append(expected_observation)

                    action = actions1[:, s+t:s+singlestep+t]
                    action = np.reshape(action, [num_env*singlestep])
                    actions.append(action)

                    mask = mask1[:, s+t:s+singlestep+t]
                    mask = np.reshape(mask, [num_env*singlestep])
                    masks.append(mask)

                if s > 0:
                    loss, prediction, hidden_states = env_model.train_and_predict(input_frames, actions, masks, expected_observations, hidden_states)
                    total_loss += loss
                else:
                    # warm up
                    prediction, hidden_states = env_model.predict(input_frames, actions, masks, hidden_states)


                if visualize and s == 3 * singlestep:
                    for batch_idx in range(num_env*singlestep):
                        expected_t = expected_observations[0]
                        if np.sum(expected_t[batch_idx,:,:,:] > 0.0):
                            input_frame = input_frames[batch_idx, :, :, :]
                            cv2.imshow('input', input_frame)
                            for i in range(K):
                                time_t_expectation = expected_observations[i]
                                exp_obs = time_t_expectation[batch_idx,:,:,:]
                                cv2.imshow('expected for t+{}'.format(i+1), exp_obs)
                            for i in range(K):
                                time_t_prediction = prediction[i]
                                cv2.imshow('prediction for t+{}'.format(i+1), time_t_prediction[batch_idx, :,:,7])
                            cv2.waitKey(0)

        print("avg_loss = {}".format(total_loss / K / save_freq / valid_batch_size))
        env_model.save()

if __name__ == '__main__':
    main(visualize=False)