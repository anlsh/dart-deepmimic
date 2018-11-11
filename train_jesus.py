from baselines.common.cmd_util import make_dart_env, common_arg_parser
from baselines.common import tf_util as U
from baselines import logger

from Humanoid3D_cartesian_jesus import DartHumanoid3D_cartesian_jesus
from baselines.ppo1 import mlp_policy, pposgd_simple

def make_dart_env(seed):
    env = DartHumanoid3D_cartesian_jesus(seed)
    env = Monitor(env, logger.get_dir())
    return env

def policy_fn(name, ob_space, ac_space):
    # TODO Ensure that multiple-layers implementation is really solid
    return mlp_policy.MlpPolicy(name=name,
                                ob_space=ob_space, ac_space=ac_space,
                                hidden_dimension_list=[128,128])

def train(num_timesteps, seed,
          train_save_interval, output_prefix, ):

    U.make_session(num_cpu=1).__enter__()
    env = make_dart_env(seed)

    def callback_fn(local_vars, global_vars):
        iters = local_vars["iters_so_far"]
        saver = tf.train.Saver()
        if iters % save_interval == 0:
            saver.save(sess, output_prefix + str(iters))


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

    ######################################################################
    # MY CUSTOM ARGS


    parser.add_argument('--train-save-interval', type=int, default=100,
                        help="Interval between saves and stuff")
    parser.add_argument('--output-prefix', required=True,
                        help="Fire prefix of parameter saves")

    # TODO Disabled for now!!! CPU thing isn't critical though
    # parser.add_argument('--num-cpus', type=int, default=1,
    #                     help="Number of CPU cores to use? Idk...")
    # parser.add_argument('--hidden-dims', type=str, default="64,64",
    #                     help="Within quotes, sizes of each hidden layer "
    #                     + "seperated by commas [also, no whitespace]")

    # END CUSTOM ARGS
    ######################################################################


    args = parser.parse_args()
    logger.configure()

    train(num_timesteps=args.num_timesteps, seed=args.seed,
          train_save_interval=args.train_save_interval,
          output_prefix=args.output_prefix)

if __name__ == '__main__':
    main()
