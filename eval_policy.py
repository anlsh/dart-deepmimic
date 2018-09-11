import argparse
from visak_dartdeepmimic import VisakDartDeepMimicArgParse
from baselines.common import tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple
import numpy as np
import tensorflow as tf

class PolicyLoaderAgent(object):
    """The world's simplest agent!"""
    def __init__(self, param_path, obs_space, action_space):
        self.action_space = action_space

        self.actor = mlp_policy.MlpPolicy("pi", obs_space, action_space,
                                          hid_size = 64, num_hid_layers=2)
        U.initialize()
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), param_path)

    def act(self, observation, reward, done):
        action2, unknown = self.actor.act(False, observation)
        return action2


if __name__ == "__main__":

    parser = VisakDartDeepMimicArgParse()
    parser.add_argument("--params-prefix", required=True, type=str)
    args = parser.parse_args()
    env = parser.get_env()

    U.make_session(num_cpu=1).__enter__()

    U.initialize()

    agent = PolicyLoaderAgent(args.params_prefix, env.observation_space, env.action_space)

    episode_count = 100
    reward = 0
    done = False

    while True:
        ob = env.reset(0, pos_stdv=0, vel_stdv=0)
        done = False
        while not done:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            env.render()
