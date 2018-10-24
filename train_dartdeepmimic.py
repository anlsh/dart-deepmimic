from baselines.common import set_global_seeds, tf_util as U
import tensorflow as tf
from baselines import bench
import os.path as osp
import gym, logging
from baselines.bench import Monitor
from baselines import logger

from dartdeepmimic import DartDeepMimicEnv
from ddm_argparse import DartDeepMimicArgParse

def train(env, initial_params_path,
          save_interval, out_prefix, num_timesteps, num_cpus,
          hidden_dimensions):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=num_cpus).__enter__()

    U.initialize()

    def policy_fn(name, ob_space, ac_space):
        print("Policy with name: ", name)
        policy = mlp_policy.MlpPolicy(name=name, ob_space=ob_space,
                                      ac_space=ac_space,
                                      hidden_dimension_list=hidden_dimensions)
        saver = tf.train.Saver()
        if initial_params_path is not None:
            saver.restore(sess, initial_params_path)
        return policy

    #env = bench.Monitor(env, "results.json")
    # env.seed(8)
    set_global_seeds(8)
    gym.logger.setLevel(logging.WARN)

    def callback_fn(local_vars, global_vars):
        iters = local_vars["iters_so_far"]
        if iters == 0 and initial_params_path is not None:
            print("Restoring from " + initial_params_path)
            tf.train.Saver().restore(tf.get_default_session(),
                                     initial_params_path)
        saver = tf.train.Saver()
        if iters % save_interval == 0:
            saver.save(sess, out_prefix + str(iters))

    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            callback=callback_fn,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=1.0, lam=0.95, schedule='linear',
        )
    env.close()

if __name__ == '__main__':

    parser = DartDeepMimicArgParse()
    parser.add_argument('--train-save-interval', type=int, default=100,
                        help="Interval between saves and stuff")
    parser.add_argument('--initial-params-path', type=str, default=None,
                        help="Path to the prefix of files you want to load from")
    parser.add_argument('--output-params-prefix', required=True,
                        help="Fire prefix of parameter saves")
    parser.add_argument('--num-cpus', type=int, default=1,
                        help="Number of CPU cores to use? Idk...")
    parser.add_argument('--train-num-timesteps', type=int, default=7e8,
                        help="No idea what this does")
    parser.add_argument('--hidden-dims', type=str, default="64,64",
                        help="Within quotes, sizes of each hidden layer "
                        + "seperated by commas [also, no whitespace]")

    args = parser.parse_args()
    env = parser.get_env()
    hidden_dimensions = [int(i) for i in args.hidden_dims.split(",")]
    #####################################
    # END COPY-PASTE FROM DARTDEEPMIMIC #
    #####################################

    train(env,
          initial_params_path=args.initial_params_path,
          save_interval=args.train_save_interval,
          out_prefix=args.output_params_prefix,
          num_timesteps=args.train_num_timesteps,
          num_cpus=args.num_cpus,
          hidden_dimensions=hidden_dimensions)
