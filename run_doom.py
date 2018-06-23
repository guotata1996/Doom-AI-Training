#!/usr/bin/env python3

from baselines import logger
from baselines.common.cmd_util import atari_arg_parser
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.a2c.a2c import learn
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from env.circleshootenv import ShootEnv
from env.navienv import NaviEnv
from env.mixedenv import MixedEnv
from env.conservativenavienv import ConservativeNaviEnv
from env.dodgenv import DodgeEnv
from env.upfloor import UpFloorEnv
from env.finddoor import FindDoorEnv
from env.gatherenv import GatherEnv
from model.i2aNetwork import I2ANetwork

def make_doom_env(num_env, seed, name):
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            if name == 'shoot':
                env = ShootEnv()
                env.seed(rank)
            elif name == 'navi':
                env = NaviEnv()
                env.seed(rank)
            elif name == 'consnavi':
                env = ConservativeNaviEnv()
                env.seed(rank)
            elif name == 'mixed':
                env = MixedEnv()
                env.seed(rank)
            elif name == 'dodge':
                env = DodgeEnv()
                env.seed(rank)
            elif name == 'upfloor':
                env = UpFloorEnv()
                env.seed(seed+rank)
            elif name == 'finddoor':
                env = FindDoorEnv()
                env.seed(rank)
            elif name == 'gather':
                env = GatherEnv()
                env.seed(rank)
            else:
                print('Invalid env name')
            #For finddoor env

            #env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i) for i in range(num_env)])

def train(num_timesteps, env_name, seed, policy, lrschedule, num_env, entrophy, lr, save_name = None):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy == 'i2a':
        policy_fn = I2ANetwork
    env = VecFrameStack(make_doom_env(num_env, 0, env_name), 4)
    if save_name is None:
        save_name = env_name
    learn(policy_fn, env, seed, save_name=save_name,total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule, log_interval=500, save_interval=1000, cont=True, ent_coef=entrophy, lr=lr)
    env.close()

def main():
    parser = atari_arg_parser()
    parser.add_argument('--ent', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--policy', type=str)
    parser.add_argument('--save_name', default=None, type=str)
    args = parser.parse_args()
    logger.configure()
    print('saving to:{}'.format(args.save_name))
    train(num_timesteps=110000000, env_name=args.env, seed=args.seed,
        policy=args.policy, lrschedule='constant', num_env=16, entrophy=args.ent, lr=args.lr, save_name=args.save_name)

if __name__ == '__main__':
    main()
