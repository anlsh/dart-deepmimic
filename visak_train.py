from baselines.common import set_global_seeds, tf_util as U
import tensorflow as tf
from baselines import bench
import os.path as osp
import gym, logging
from baselines.bench import Monitor
from baselines import logger
import sys
import random
from humanoid_redux import DartHumanoid3D_cartesian


def make_dart_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    print("#####################################")
    print("seed",seed)

    set_global_seeds(seed)
    env = DartHumanoid3D_cartesian()
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env


def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    #with tf.variable_scope('inclined'):
    U.make_session(num_cpu=1).__enter__()
    #set_global_seeds(seed)
    seed = random.randint(1,10)
    env = make_dart_env(env_id,seed)#gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hidden_dimension_list=[64,64])
    #env = bench.Monitor(env, "results.json")
    #env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=1.0, lam=0.95, schedule='linear',
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHumanoid3dPIDWalk-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=8)
    args = parser.parse_args()
    train(args.env, num_timesteps=7e8, seed=args.seed)


if __name__ == '__main__':
    main()
