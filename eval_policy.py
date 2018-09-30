import argparse
import ddm_argparse
from baselines.common import tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple
import numpy as np
import tensorflow as tf
import random

class PolicyLoaderAgent(object):
    """The world's simplest agent!"""
    def __init__(self, param_path, obs_space, action_space, hidden_dims):
        self.action_space = action_space

        self.actor = mlp_policy.MlpPolicy("pi", obs_space, action_space,
                                          hidden_dimension_list=hidden_dims)
        U.initialize()
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), param_path)

    def act(self, observation, reward, done):
        action2, unknown = self.actor.act(False, observation)
        return action2


if __name__ == "__main__":

    parser = ddm_argparse.DartDeepMimicArgParse()
    parser.add_argument("--params-prefix", required=True, type=str)
    parser.add_argument('--hidden-dims', type=str, default="64,64",
                        help="Within quotes, sizes of each hidden layer "
                        + "seperated by commas [also, no whitespace]")
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
    hidden_dims = [int(i) for i in args.hidden_dims.split(",")]
    env = parser.get_env()

    U.make_session(num_cpu=1).__enter__()

    U.initialize()

    agent = PolicyLoaderAgent(args.params_prefix,
                              env.observation_space,
                              env.action_space,
                              hidden_dims=hidden_dims)

    episode_count = 100
    reward = 0
    done = False

    while True:
        if not args.randinit:
            ob = env.reset(0, pos_stdv=0, vel_stdv=0)
        else:
            ob = env.reset(random.randint(0, env.num_frames - 1),
                           pos_stdv=0, vel_stdv=0)

        done = False
        while (not done) if args.terminate else True:
            if env.framenum == env.num_frames - 1:
                env.framenum = 0
            action = agent.act(ob, reward, done)
            reward = env.reward(env.control_skel, env.framenum)
            ob, reward, done, _ = env.step(action)
            env.render()
