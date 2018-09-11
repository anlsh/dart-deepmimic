from baselines.common import tf_util as U
import tensorflow as tf
import gym, logging

from visak_dartdeepmimic import VisakDartDeepMimicArgParse

def train(env, initial_params_path,
          save_interval, out_prefix, num_timesteps, num_cpus):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=num_cpus).__enter__()

    U.initialize()

    def policy_fn(name, ob_space, ac_space):
        print("Policy with name: ", name)
        policy = mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
        saver = tf.train.Saver()
        if initial_params_path is not None:
            print("Tried to restore from ", initial_params_path)
            saver.restore(sess, initial_params_path)
        return policy

    gym.logger.setLevel(logging.WARN)

    def callback_fn(local_vars, global_vars):
        iters = local_vars["iters_so_far"]
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

    parser = VisakDartDeepMimicArgParse()
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

    args = parser.parse_args()
    env = parser.get_env()
    #####################################
    # END COPY-PASTE FROM DARTDEEPMIMIC #
    #####################################

    train(env,
          initial_params_path=args.initial_params_path,
          save_interval=args.train_save_interval,
          out_prefix=args.output_params_prefix,
          num_timesteps=args.train_num_timesteps,
          num_cpus=args.num_cpus)
