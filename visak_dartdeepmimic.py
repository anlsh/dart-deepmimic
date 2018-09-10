from dartdeepmimic import DartDeepMimicEnv, map_dofs, END_OFFSET
from ddm_argparse import DartDeepMimicArgParse
import numpy as np
import argparse
from runner import EnvPlayer
from copy import copy

class VisakDartDeepMimicEnv(DartDeepMimicEnv):

    def construct_frames(self, raw_framelist):
        """
        AMC data is given in sequential degrees, while dart specifies angles
        in rotating radians. The conversion is quite expensive, so we precomute
        all positions and velocities and store the results
        """

        raw_framelist = None

        with open("visak-stuff/JustJumpPositions_corrected.txt","rb") as fp:
            WalkPositions = np.loadtxt(fp)
        with open("visak-stuff/JustJumpVelocities_corrected.txt","rb") as fp:
            WalkVelocities = np.loadtxt(fp)

        num_frames = len(WalkPositions)

        pos_frames = [None] * num_frames
        vel_frames = [None] * num_frames
        quat_frames = [None] * num_frames
        com_frames = [None] * num_frames
        ee_frames = [None] * num_frames


        for i in range(len(WalkPositions)):

            # TODO EMERGENCY Remember to the change the root angles to
            # zyx order

            updated_pos = WalkPositions[i,:].copy()
            updated_pos[3:6] = updated_pos[3:6][::-1]
            temp = updated_pos[3:6].copy()
            updated_pos[3:6] = updated_pos[0:3]
            updated_pos[0:3] = temp

            updated_vel = WalkVelocities[i,:].copy()
            updated_vel[3:6] = updated_vel[3:6][::-1]
            temp = updated_vel[3:6].copy()
            updated_vel[3:6] = updated_pos[0:3]
            updated_vel[0:3] = temp

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

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = 5.0
            self._get_viewer().scene.tb.trans[2] = -7.5
            self._get_viewer().scene.tb.trans[1] = 0.0
            #-10.0


class VisakDartDeepMimicArgParse(DartDeepMimicArgParse):

    def get_env(self):

        return VisakDartDeepMimicEnv(control_skeleton_path=self.args.control_skel_path,
                                     reference_motion_path=self.args.ref_motion_path,
                                     refmotion_dt=self.args.ref_motion_dt,
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
