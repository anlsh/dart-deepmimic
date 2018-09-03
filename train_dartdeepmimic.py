from baselines.common import set_global_seeds, tf_util as U
import tensorflow as tf
from baselines import bench
import os.path as osp
import gym, logging
from baselines.bench import Monitor
from baselines import logger
import argparse

from dartdeepmimic import DartDeepMimicEnv

def train(env, save_interval, file_prefix, num_timesteps):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    #with tf.variable_scope('inclined'):
    U.make_session(num_cpu=1).__enter__()
    #set_global_seeds(seed)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    #env = bench.Monitor(env, "results.json")
    #env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    def callback_fn(local_vars, global_vars):
        # TODO Implement proper handling of writing to files and stuff
        # TODO also probably dont save on every single iteration...
        iters = local_vars["iters_so_far"]
        if iters % save_interval == 0:
            U.save_state(file_prefix + str(iters))

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

    parser = argparse.ArgumentParser(description='Make a DartDeepMimic Environ')
    parser.add_argument('--train-save-interval', type=int, default=25,
                        help="Interval between saves and stuff")
    parser.add_argument('--train-params-prefix', required=True,
                        help="Fire prefix of parameter saves")
    parser.add_argument('--train-num-timesteps', type=int, default=7e8,
                        help="No idea what this does")


    #######################################
    # BEGIN COPY-PASTE FROM DARTDEEPMIMIC #
    #######################################

    parser.add_argument('--control-skel-path', required=True,
                        help='Path to the control skeleton')
    parser.add_argument('--asf-path', required=True,
                        help='Path to asf which the skeleton was parsed from')
    parser.add_argument('--ref-motion-path', required=True,
                        help='Path to the reference motion AMC')
    parser.add_argument('--state-mode', default=0, type=int,
                        help="Code for the state representation")
    parser.add_argument('--action-mode', default=0, type=int,
                        help="Code for the action representation")
    parser.add_argument('--visualize', default=False,
                        help="DOESN'T DO ANYTHING RIGHT NOW: True if you want a window to render to")
    parser.add_argument('--max-torque', type=float, default=90,
                        help="Maximum torque")
    parser.add_argument('--default-damping', type=float, default=80,
                        help="Default damping coefficient for joints")
    parser.add_argument('--default-spring', type=float, default=0,
                        help="Default spring stiffness for joints")
    parser.add_argument('--simsteps-per-dataframe', type=int, default=10,
                        help="Number of simulation steps per frame of mocap" +
                        " data")
    parser.add_argument('--reward-cutoff', type=float, default=0.1,
                        help="Terminate the episode when rewards below this " +
                             "threshold are calculated. Should be in range (0, 1)")
    parser.add_argument('--window-width', type=int, default=80,
                        help="Window width")
    parser.add_argument('--window-height', type=int, default=45,
                        help="Window height")

    parser.add_argument('--pos-init-noise', type=float, default=.05,
                        help="Standard deviation of the position init noise")
    parser.add_argument('--vel-init-noise', type=float, default=.05,
                        help="Standart deviation of the velocity init noise")

    parser.add_argument('--pos-weight', type=float, default=.65,
                        help="Weighting for the pos difference in the reward")
    parser.add_argument('--pos-inner-weight', type=float, default=-2,
                        help="Coefficient for pos difference exponentiation in reward")

    parser.add_argument('--vel-weight', type=float, default=.1,
                        help="Weighting for the pos difference in the reward")
    parser.add_argument('--vel-inner-weight', type=float, default=-.1,
                        help="Coefficient for vel difference exponentiation in reward")

    parser.add_argument('--ee-weight', type=float, default=.15,
                        help="Weighting for the pos difference in the reward")
    parser.add_argument('--ee-inner-weight', type=float, default=-40,
                        help="Coefficient for pos difference exponentiation in reward")

    parser.add_argument('--com-weight', type=float, default=.1,
                        help="Weighting for the com difference in the reward")
    parser.add_argument('--com-inner-weight', type=float, default=-10,
                        help="Coefficient for com difference exponentiation in reward")

    parser.add_argument('--p-gain', type=float, default=300,
                        help="P for the PD controller")
    parser.add_argument('--d-gain', type=float, default=50,
                        help="D for the PD controller")

    args = parser.parse_args()

    env = DartDeepMimicEnv(args.control_skel_path, args.asf_path,
                           args.ref_motion_path,
                           args.state_mode, args.action_mode,
                           args.p_gain, args.d_gain,
                           args.pos_init_noise, args.vel_init_noise,
                           args.reward_cutoff,
                           args.pos_weight, args.pos_inner_weight,
                           args.vel_weight, args.vel_inner_weight,
                           args.ee_weight, args.ee_inner_weight,
                           args.com_weight, args.com_inner_weight,
                           args.max_torque,
                           args.default_damping, args.default_spring,
                           args.visualize,
                           args.simsteps_per_dataframe,
                           args.window_width, args.window_height)

    #####################################
    # END COPY-PASTE FROM DARTDEEPMIMIC #
    #####################################

    train(env, args.train_save_interval, args.train_params_prefix,
          args.train_num_timesteps)
