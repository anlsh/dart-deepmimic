import argparse
from dartdeepmimic import DartDeepMimicArgParse
from baselines.common import tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple
import numpy as np

class PolicyLoaderAgent(object):
    """The world's simplest agent!"""
    def __init__(self, param_path, obs_space, action_space):
        self.action_space = action_space

        # TODO These parameters are COUPLED: should match up with those in
        # run_dart.py which is over in baselines
        self.actor = mlp_policy.MlpPolicy("pi", obs_space, action_space,
                                          hid_size = 64, num_hid_layers=2)
        U.initialize()
        U.load_state(param_path)

    def act(self, observation, reward, done):
        # action1 = self.action_space.sample()
        action2, unknown = self.actor.act(False, observation)
        return action2


if __name__ == "__main__":

    parser = DartDeepMimicArgParse()
    parser.add_argument("--params-prefix", required=True, type=str)
    args = parser.parse_args()
    env = parser.get_env()

    env = parser.get_env()

    U.make_session(num_cpu=1).__enter__()

    U.initialize()

    agent = PolicyLoaderAgent(args.params_prefix, env.observation_space, env.action_space)

    episode_count = 100
    reward = 0
    done = False

    while True:
        ob = env.reset(pos_stdv=0, vel_stdv=0)
        done = False
        while not done:
            action = np.multiply(agent.act(ob, reward, done), 0)
            ob, reward, done, _ = env.step(action)
            env.render()


    # for i in range(episode_count):
    #     ob = env.reset()
    #     while True:
    #         action = agent.act(ob, reward, done)
    #         ob, reward, done, _ = env.step(action)
    #         if done:
    #             break
    #         # Note there's no env.render() here. But the environment still can open window and
    #         # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
    #         # Video is not recorded every episode, see capped_cubic_video_schedule for details.
