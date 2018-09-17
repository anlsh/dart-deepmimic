import ddm_argparse
import numpy as np

class EnvPlayer:

    def __init__(self, env):

        self.env = env

    def play_motion_no_noise(self):
        for i in range(self.env.num_frames):
            obs = self.env.reset(i, 0, 0)
            print(env.should_terminate(0, obs))
            self.env.render()

    def take_single_step_zero_pos(self, init_frame=0):
        obs = self.env.reset(init_frame, 0, 0)
        self.env.step(np.zeros(self.env.action_dim))

    def init_to_frame_passive(self, init_frame=0):
        obs = self.env.reset(init_frame, 0, 0)
        while True:
            env.do_simulation(np.zeros(len(env.control_skel.q)),
                              env.simsteps_per_dataframe)
            env.render()

if __name__ == "__main__":

    # Don't run this as main, there's really not too much point

    parser = ddm_argparse.DartDeepMimicArgParse()
    args = parser.parse_args()
    env = parser.get_env()

    player = EnvPlayer(env)

    player.play_motion_no_noise()

    # env.reset(0, pos_stdv=args.pos_init_noise, vel_stdv=args.vel_init_noise)
    # done = False
    # i = 0
    # rewards = []
    # for i in range(len(env.ref_com_frames) -5 ):
    #     env.render()
    #     env.sync_skel_to_frame(env.control_skel, i, args.pos_init_noise,
    #                             args.vel_init_noise)
    #     for k in range(10):
    #         rewards.append(env.reward(env.control_skel, i))
    #         env.do_simulation(np.zeros(56), 10)
    #     i += 1

    # print(min(rewards))

    # env.reset(0, False)
    # env.reward(env.control_skel, 0)

    # PID Test stuff
    # start_frame = 0
    # target_frame = 200
    # env.sync_skel_to_frame(env.control_skel, target_frame, 0, 0)

    # # print("Provided Target Q: \n", env.control_skel.q[6:])
    # target_state = env.posveltuple_as_trans_plus_eulerlist(env.control_skel)
    # pos, vel = target_state
    # target_angles = pos[1][1:]
    # # print("Provided Target Angles\n", target_angles)

    # obs = env.sync_skel_to_frame(env.control_skel, start_frame, 0, 0)
    # print(env.control_skel.dq)

    # while True:
    #     env.framenum = target_frame
    #     s, r, done, info = env.step(np.concatenate(target_angles))
    #     env.render()

    # frame = 0
    # env.sync_skel_to_frame(env.control_skel, 0, 0, 0)
    # while True:
    #     env.render()
