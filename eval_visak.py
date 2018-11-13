import argparse
import ddm_argparse
from baselines.common import tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple
import numpy as np
import tensorflow as tf
import random
from test_ddm import vddm_env
import argparse
from gym.envs.registration import register

class PolicyLoaderAgent(object):
    """The world's simplest agent!"""
    def __init__(self, param_path, obs_space, action_space, hid_size,
                 num_hid_layers):
        self.action_space = action_space

        self.actor = mlp_policy.MlpPolicy("pi", obs_space, action_space,
                                          hid_size=hid_size,
                                          num_hid_layers=num_hid_layers)
        U.initialize()
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), param_path)

    def act(self, observation, reward, done):
        action2, unknown = self.actor.act(False, observation)
        return action2


if __name__ == "__main__":

    # parser = ddm_argparse.DartDeepMimicArgParse()
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-prefix", required=True, type=str)
    # parser.add_argument('--hidden-dims', type=str, default="64,64",
    #                     help="Within quotes, sizes of each hidden layer "
    #                     + "seperated by commas [also, no whitespace]")
    parser.add_argument('--hid-size', default=64, type=int)
    parser.add_argument('--num-hid-layers', default=2, type=int)
    terminate_group = parser.add_mutually_exclusive_group()
    terminate_group.add_argument('--use-env-done',
                                dest='terminate',
                                action='store_true')
    terminate_group.add_argument('--no-use-env-done',
                                dest='terminate',
                                action='store_false')
    parser.set_defaults(terminate=True,
                        help="Whether to enable gravity in the world")
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument('--init-from-start',
                            dest='randinit',
                            action='store_false')
    init_group.add_argument('--no-init-from-start',
                            dest='randinit',
                            action='store_true')
    parser.set_defaults(randinit=True,
                        help="Whether to initialize from start or randomly")

    args = parser.parse_args()
    env = vddm_env(random.randint(0, 30000))

    U.make_session(num_cpu=1).__enter__()

    U.initialize()

    agent = PolicyLoaderAgent(args.params_prefix,
                              env.observation_space,
                              env.action_space,
                              hid_size=args.hid_size,
                              num_hid_layers=args.num_hid_layers)

    episode_count = 100
    reward = 0
    done = False

    while True:
        if not args.randinit:
            # ob = env.reset(0, pos_stdv=0, vel_stdv=0)
            # ob = env.reset()
            env.framenum = 0
            qpos = env.MotionPositions[env.framenum,
                                        :].reshape(env.robot_skeleton.ndofs) \
                + env.np_random.uniform(low=0, high=0,
                                        size=env.robot_skeleton.ndofs)

            qvel = env.MotionVelocities[env.framenum,
                                        :].reshape(env.robot_skeleton.ndofs) \
                + env.np_random.uniform(low=-0, high=0,
                                        size=env.robot_skeleton.ndofs)

            env.set_state(qpos, qvel)

            ob = env._get_obs()

        else:
            # ob = env.reset(random.randint(0, env.num_frames - 1),
            #                pos_stdv=0, vel_stdv=0)
            ob = env.reset(random.randint(0, env.num_frames - 1))

        done = False
        cum_reward = 0
        length = 0
        while (not done) if args.terminate else True:
            if env.framenum == env.num_frames - 1:
                env.framenum = 0
            action = agent.act(ob, reward, done)
            # reward = env.reward(env.robot_skeleton, env.framenum)
            ob, reward, done, _ = env.step(action)
            cum_reward += reward
            length += 1
            env.render("human")

        print("Total-Reward, Length = " + str((cum_reward, length)))
