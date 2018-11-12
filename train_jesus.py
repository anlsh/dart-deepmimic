from baselines.common.cmd_util import common_arg_parser
from baselines.common import tf_util as U
import tensorflow as tf
from baselines import logger
from baselines.bench import Monitor
import gym
from baselines.common import set_global_seeds, tf_util as U
from gym.envs.registration import register

register(
    id='raw-v0',
    entry_point='env_jesus:DartHumanoid3D_cartesian',
)

def make_dart_env(env_id, seed):
    print("#####################################")
    print("seed",seed)
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    return env

def train(env_id, num_timesteps, seed,
          save_interval, output_prefix):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=1)
    sess.__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=128, num_hid_layers=2)

    def callback_fn(local_vars, global_vars):
        iters = local_vars["iters_so_far"]
        saver = tf.train.Saver()
        if iters % save_interval == 0:
            saver.save(sess, output_prefix + str(iters))

    env = make_dart_env(env_id, seed)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
                        callback=callback_fn
        )
    env.close()

def main():
    parser = common_arg_parser()
    parser.add_argument('--save-interval', type=int, default=100,
                        help="Interval between saves and stuff")
    parser.add_argument('--output-prefix', required=True,
                        help="Fire prefix of parameter saves")

    args = parser.parse_args()

    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          save_interval=args.save_interval,
          output_prefix=args.output_prefix)

if __name__ == '__main__':
    main()
