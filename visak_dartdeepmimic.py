from dartdeepmimic import DartDeepMimicEnv, map_dofs, END_OFFSET
import numpy as np
import argparse

class VisakDartDeepMimicEnv(DartDeepMimicEnv):

    def construct_frames(self, raw_framelist):
        """
        AMC data is given in sequential degrees, while dart specifies angles
        in rotating radians. The conversion is quite expensive, so we precomute
        all positions and velocities and store the results
        """

        num_frames = len(raw_framelist)
        elements_per_frame = len(raw_framelist[0])

        pos_frames = [None] * num_frames
        vel_frames = [None] * num_frames
        quat_frames = [None] * num_frames
        com_frames = [None] * num_frames
        ee_frames = [None] * num_frames

        with open("visak-stuff/JustJumpPositions_corrected.txt","rb") as fp:
            WalkPositions = np.loadtxt(fp)
        with open("visak-stuff/JustJumpVelocities_corrected.txt","rb") as fp:
            WalkVelocities = np.loadtxt(fp)

        for i in range(len(WalkPositions)):

            # TODO EMERGENCY Remember to the change the root angles to
            # zyx order

            updated_pos = WalkPositions[i,:]
            updated_pos[3:6] = updated_pos[3:6][::-1]
            updated_vel = WalkVelocities[i,:]
            updated_vel[3:6] = updated_vel[3:6][::-1]

            map_dofs(self.ref_skel.dofs, updated_pos,
                     updated_vel, 0, 0)
            pos_frames[i] = updated_pos
            vel_frames[i] = updated_vel
            com_frames[i] = self.ref_skel.com()
            quat_frames[i] = self.quaternion_angles(self.ref_skel)
            # TODO Parse actual end positions
            ee_frames[i] = [self.ref_skel.bodynodes[ii].to_world(END_OFFSET)
                            for ii in self._end_effector_indices]

        return num_frames, (pos_frames, vel_frames, quat_frames, com_frames,
                            ee_frames)


class VisakDartDeepMimicArgParse(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()
        self.add_argument('--control-skel-path', required=True,
                          help='Path to the control skeleton')
        self.add_argument('--ref-motion-path', required=True,
                          help='Path to the reference motion AMC')
        self.add_argument('--state-mode', default=0, type=int,
                          help="Code for the state representation")
        self.add_argument('--action-mode', type=int, required=True,
                          help="Code for the action representation")
        self.add_argument('--visualize', default=False,
                          help="DOESN'T DO ANYTHING RIGHT NOW: True if you want"
                          + " a window to render to")
        self.add_argument('--max-torque', type=float, default=90,
                          help="Maximum torque")
        self.add_argument('--max-angle', type=float, default=5,
                          help="Max magnitude of angle (in terms of pi) that "
                          + "PID can output")
        self.add_argument('--default-damping', type=float, default=80,
                          help="Default damping coefficient for joints")
        self.add_argument('--default-spring', type=float, default=0,
                          help="Default spring stiffness for joints")
        self.add_argument('--simsteps-per-dataframe', type=int, default=10,
                          help="Number of simulation steps per frame of mocap" +
                          " data")
        self.add_argument('--reward-cutoff', type=float, default=0.1,
                          help="Terminate the episode when rewards below this" +
                          " threshold are calculated. Should be in range (0, 1)")
        self.add_argument('--window-width', type=int, default=80,
                          help="Window width")
        self.add_argument('--window-height', type=int, default=45,
                          help="Window height")

        self.add_argument('--pos-init-noise', type=float, default=.05,
                          help="Standard deviation of the position init noise")
        self.add_argument('--vel-init-noise', type=float, default=.05,
                          help="Standart deviation of the velocity init noise")

        self.add_argument('--pos-weight', type=float, default=.65,
                          help="Weighting for the pos difference in the reward")
        self.add_argument('--pos-inner-weight', type=float, default=-2,
                          help="Coefficient for pos difference exponentiation in reward")

        self.add_argument('--vel-weight', type=float, default=.1,
                          help="Weighting for the pos difference in the reward")
        self.add_argument('--vel-inner-weight', type=float, default=-.1,
                          help="Coefficient for vel difference exponentiation in reward")

        self.add_argument('--ee-weight', type=float, default=.15,
                          help="Weighting for the pos difference in the reward")
        self.add_argument('--ee-inner-weight', type=float, default=-40,
                          help="Coefficient for pos difference exponentiation in reward")

        self.add_argument('--com-weight', type=float, default=.1,
                          help="Weighting for the com difference in the reward")
        self.add_argument('--com-inner-weight', type=float, default=-10,
                          help="Coefficient for com difference exponentiation in reward")

        self.add_argument('--p-gain', type=float, default=300,
                            help="P for the PD controller")
        self.add_argument('--d-gain', type=float, default=50,
                          help="D for the PD controller")

        gravity_group = self.add_mutually_exclusive_group()
        gravity_group.add_argument('--gravity',
                                   dest='gravity',
                                   action='store_true')
        gravity_group.add_argument('--no-gravity',
                                   dest='gravity',
                                   action='store_false')
        self.set_defaults(gravity=True, help="Whether to enable gravity in the world")

        self_collide_group = self.add_mutually_exclusive_group()
        self_collide_group.add_argument('--self-collide',
                                        dest='selfcollide',
                                        action='store_true')
        self_collide_group.add_argument('--no-self-collide',
                                        dest='selfcollide',
                                        action='store_false')
        self.set_defaults(self_collide=True, help="Whether to enable selfcollisions in the skeleton")

        self.args = None

    def parse_args(self):
        self.args = super().parse_args()
        return self.args

    def get_env(self):

        return VisakDartDeepMimicEnv(control_skeleton_path=self.args.control_skel_path,
                                reference_motion_path=self.args.ref_motion_path,
                                statemode=self.args.state_mode,
                                actionmode=self.args.action_mode,
                                p_gain=self.args.p_gain,
                                d_gain=self.args.d_gain,
                                pos_init_noise=self.args.pos_init_noise,
                                vel_init_noise=self.args.vel_init_noise,
                                reward_cutoff=self.args.reward_cutoff,
                                pos_weight=self.args.pos_weight,
                                pos_inner_weight=self.args.pos_inner_weight,
                                vel_weight=self.args.vel_weight,
                                vel_inner_weight=self.args.vel_inner_weight,
                                ee_weight=self.args.ee_weight,
                                ee_inner_weight=self.args.ee_inner_weight,
                                com_weight=self.args.com_weight,
                                com_inner_weight=self.args.com_inner_weight,
                                max_torque=self.args.max_torque,
                                max_angle=self.args.max_angle,
                                default_damping=self.args.default_damping,
                                default_spring=self.args.default_spring,
                                visualize=self.args.visualize,
                                simsteps_per_dataframe=self.args.simsteps_per_dataframe,
                                screen_width=self.args.window_width,
                                screen_height=self.args.window_height,
                                gravity=self.args.gravity,
                                self_collide=self.args.selfcollide)

if __name__ == "__main__":

    parser = VisakDartDeepMimicArgParse()
    args = parser.parse_args()
    env = parser.get_env()

    env.reset(0, pos_stdv=args.pos_init_noise, vel_stdv=args.vel_init_noise)
    done = False
    i = 0
    rewards = []
    for i in range(len(env.ref_com_frames) -5 ):
        # env.render()
        env.sync_skel_to_frame(env.control_skel, i, args.pos_init_noise,
                                args.vel_init_noise)
        for k in range(10):
            rewards.append(env.reward(env.control_skel, i))
            env.do_simulation(np.zeros(56), 10)
        i += 1

    print(min(rewards))

    # for i in range(env.num_frames):
    #     env.reset(i, False)
    #     env.render()
    #     env.reward(env.control_skel, i)
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
