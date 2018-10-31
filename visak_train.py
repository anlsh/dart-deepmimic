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


def make_dart_env(seed):
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


def train(num_timesteps, seed,
          out_prefix,
          save_interval, num_cpus,
          hidden_dims=[64,64]):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    #with tf.variable_scope('inclined'):
    sess = U.make_session(num_cpu=num_cpus)
    sess.__enter__()
    #set_global_seeds(seed)
    seed = random.randint(1,10)
    env = make_dart_env(seed)#gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name,
                                    ob_space=ob_space, ac_space=ac_space,
                                    hidden_dimension_list=hidden_dims)

    def callback_fn(local_vars, global_vars):
        iters = local_vars["iters_so_far"]
        saver = tf.train.Saver()
        if iters % save_interval == 0:
            saver.save(sess, out_prefix + str(iters))

    #env = bench.Monitor(env, "results.json")
    #env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=1.0, lam=0.95, schedule='linear',
            callback=callback_fn
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--env', help='environment ID', default='DartHumanoid3dPIDWalk-v1')
    parser.add_argument('--train-save-interval', type=int, default=100,
                        help="Interval between saves and stuff")
    # parser.add_argument('--initial-params-path', type=str, default=None,
    #                     help="Path to the prefix of files you want to load from")
    parser.add_argument('--output-params-prefix', required=True,
                        help="Fire prefix of parameter saves")
    parser.add_argument('--num-cpus', type=int, default=1,
                        help="Number of CPU cores to use? Idk...")
    parser.add_argument('--train-num-timesteps', type=int, default=7e8,
                        help="No idea what this does")
    parser.add_argument('--hidden-dims', type=str, default="64,64",
                        help="Within quotes, sizes of each hidden layer "
                        + "seperated by commas [also, no whitespace]")
    parser.add_argument('--seed', help='RNG seed', type=int, default=8)
    args = parser.parse_args()
    train(num_timesteps=args.train_num_timesteps,
          seed=args.seed,
          out_prefix=args.output_params_prefix,
          save_interval=args.train_save_interval,
          num_cpus=args.num_cpus,
          hidden_dims=[int(i) for i in args.hidden_dims.split(",")])


if __name__ == '__main__':
    main()
