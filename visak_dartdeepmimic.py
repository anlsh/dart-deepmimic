from dartdeepmimic import DartDeepMimicEnv, map_dofs, END_OFFSET
import ddm_argparse
import numpy as np
import argparse
import runner
from copy import copy


class VisakDartDeepMimicEnv(DartDeepMimicEnv):

    def __init__(self, *args, **kwargs):

        self.WalkPositions = None
        self.WalkVelocities = None

        super(VisakDartDeepMimicEnv, self).__init__(*args, **kwargs)


    def construct_frames(self, ref_motion_path):
        """
        AMC data is given in sequential degrees, while dart specifies angles
        in rotating radians. The conversion is quite expensive, so we precomute
        all positions and velocities and store the results
        """

        raw_framelist = None

        with open("assets/mocap/WalkPositions_corrected.txt","rb") as fp:
            self.WalkPositions = np.loadtxt(fp)
        with open("assets/mocap/WalkVelocities_corrected.txt","rb") as fp:
            self.WalkVelocities = np.loadtxt(fp)

        num_frames = len(self.WalkPositions)

        pos_frames = [None] * num_frames
        vel_frames = [None] * num_frames
        quat_frames = [None] * num_frames
        com_frames = [None] * num_frames
        ee_frames = [None] * num_frames


        for i in range(len(self.WalkPositions)):

            updated_pos = self.WalkPositions[i,:].copy()
            updated_pos[3:6] = updated_pos[3:6][::-1]
            temp = updated_pos[3:6].copy()
            updated_pos[3:6] = updated_pos[0:3][::-1]
            updated_pos[0:3] = temp

            updated_vel = self.WalkVelocities[i,:].copy()
            updated_vel[3:6] = updated_vel[3:6][::-1]
            temp = updated_vel[3:6].copy()
            updated_vel[3:6] = updated_pos[0:3][::-1]
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

    def should_terminate(self, reward, newstate):

        done = self.framenum == self.num_frames - 1
        done = done or not ((np.abs(newstate[2:]) < 200).all()
                            and (self.robot_skeleton.bodynodes[0].com()[1] > -0.7)
                            and (self.control_skel.q[3] > -0.4)
                            # and (self.control_skel.q[3] < 0.3)
                            and (abs(self.control_skel.q[4]) < 0.30)
                            and (abs(self.control_skel.q[5]) < 0.30))
        return done

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = 0.0
            self._get_viewer().scene.tb.trans[2] = -7.5
            self._get_viewer().scene.tb.trans[1] = 0.0

    def vsk_obs(self):

        phi = np.array([self.framenum / 322.])
        # observation for left leg thigh##################################################
        RelPos_lthigh = self.control_skel.bodynodes[2].com() - self.control_skel.bodynodes[0].com()
        state = copy.deepcopy(RelPos_lthigh)
        quat_lthigh = euler2quat(z=self.control_skel.q[8], y=self.control_skel.q[7], x=self.control_skel.q[6])
        state = np.concatenate((state, quat_lthigh))
        LinVel_lthigh = self.control_skel.bodynodes[2].dC
        state = np.concatenate((state, LinVel_lthigh))
        state = np.concatenate((state, self.control_skel.dq[6:9]))
        ################################################################3
        RelPos_lknee = self.control_skel.bodynodes[3].com() - self.control_skel.bodynodes[0].com()
        state = np.concatenate((state, RelPos_lknee))
        quat_lknee = euler2quat(z=0., y=0., x=self.control_skel.q[9])
        state = np.concatenate((state, quat_lknee))
        LinVel_lknee = self.control_skel.bodynodes[3].dC
        state = np.concatenate((state, LinVel_lknee))
        state = np.concatenate((state, np.array([self.control_skel.dq[9]])))
        #######################################################################3
        RelPos_lfoot = self.control_skel.bodynodes[4].com() - self.control_skel.bodynodes[0].com()
        state = np.concatenate((state, RelPos_lfoot))
        quat_lfoot = euler2quat(z=self.control_skel.q[11], y=0., x=self.control_skel.q[10])
        state = np.concatenate((state, quat_lfoot))
        LinVel_lfoot = self.control_skel.bodynodes[4].dC
        state = np.concatenate((state, LinVel_lfoot))
        state = np.concatenate((state, self.control_skel.dq[10:12]))
        #######################################################################3
        RelPos_rthigh = self.control_skel.bodynodes[5].com() - self.control_skel.bodynodes[0].com()
        state = np.concatenate((state, RelPos_rthigh))
        quat_rthigh = euler2quat(z=self.control_skel.q[14], y=self.control_skel.q[13], x=self.control_skel.q[12])
        state = np.concatenate((state, quat_rthigh))
        LinVel_rthigh = self.control_skel.bodynodes[5].dC
        state = np.concatenate((state, LinVel_rthigh))
        state = np.concatenate((state, self.control_skel.dq[12:15]))
        ###############################################################################3
        RelPos_rknee = self.control_skel.bodynodes[6].com() - self.control_skel.bodynodes[0].com()
        state = np.concatenate((state, RelPos_rknee))
        quat_rknee = euler2quat(z=0., y=0., x=self.control_skel.q[15])
        state = np.concatenate((state, quat_rknee))
        LinVel_rknee = self.control_skel.bodynodes[6].dC
        state = np.concatenate((state, LinVel_rknee))
        state = np.concatenate((state, np.array([self.control_skel.dq[15]])))
        ################################################################################3
        RelPos_rfoot = self.control_skel.bodynodes[7].com() - self.control_skel.bodynodes[0].com()
        state = np.concatenate((state, RelPos_rfoot))
        quat_rfoot = euler2quat(z=self.control_skel.q[17], y=0., x=self.control_skel.q[16])
        state = np.concatenate((state, quat_rfoot))
        LinVel_rfoot = self.control_skel.bodynodes[7].dC
        state = np.concatenate((state, LinVel_rfoot))
        state = np.concatenate((state, self.control_skel.dq[16:18]))
        ###########################################################
        RelPos_larm = self.control_skel.bodynodes[12].com() - self.control_skel.bodynodes[0].com()
        state = np.concatenate((state, RelPos_larm))
        quat_larm = euler2quat(z=self.control_skel.q[23], y=self.control_skel.q[22], x=self.control_skel.q[21])
        state = np.concatenate((state, quat_larm))
        LinVel_larm = self.control_skel.bodynodes[12].dC
        state = np.concatenate((state, LinVel_larm))
        state = np.concatenate((state, self.control_skel.dq[21:24]))
        ##############################################################
        RelPos_lelbow = self.control_skel.bodynodes[13].com() - self.control_skel.bodynodes[0].com()
        state = np.concatenate((state, RelPos_lelbow))
        quat_lelbow = euler2quat(z=0., y=0., x=self.control_skel.q[24])
        state = np.concatenate((state, quat_lelbow))
        LinVel_lelbow = self.control_skel.bodynodes[13].dC
        state = np.concatenate((state, LinVel_lelbow))
        state = np.concatenate((state, np.array([self.control_skel.dq[24]])))
        ################################################################
        RelPos_rarm = self.control_skel.bodynodes[15].com() - self.control_skel.bodynodes[0].com()
        state = np.concatenate((state, RelPos_rarm))
        quat_rarm = euler2quat(z=self.control_skel.q[27], y=self.control_skel.q[26], x=self.control_skel.q[25])
        state = np.concatenate((state, quat_rarm))
        LinVel_rarm = self.control_skel.bodynodes[15].dC
        state = np.concatenate((state, LinVel_rarm))
        state = np.concatenate((state, self.control_skel.dq[25:28]))
        #################################################################3
        RelPos_relbow = self.control_skel.bodynodes[16].com() - self.control_skel.bodynodes[0].com()
        state = np.concatenate((state, RelPos_relbow))
        quat_relbow = euler2quat(z=0., y=0., x=self.control_skel.q[28])
        state = np.concatenate((state, quat_relbow))
        LinVel_relbow = self.control_skel.bodynodes[16].dC
        state = np.concatenate((state, LinVel_relbow))
        state = np.concatenate((state, np.array([self.control_skel.dq[28]])))
        state = np.concatenate((state, self.control_skel.q[18:21], self.control_skel.dq[18:21], phi))
        ##################################################################

        return state


    def vsk_quatreward(self, ):
        quaternion_difference = []
        #### lthigh
        lthigh_euler = self.robot_skeleton.q[6:9]
        lthigh_mocap = self.WalkPositions[self.count, 6:9]
        quat_lthigh = euler2quat(z=lthigh_euler[2], y=lthigh_euler[1], x=lthigh_euler[0])
        quat_lthigh_mocap = euler2quat(z=lthigh_mocap[2], y=lthigh_mocap[1], x=lthigh_mocap[0])
        lthigh_diff = mult(inverse(quat_lthigh_mocap), quat_lthigh)
        scalar_lthigh = 2 * np.arccos(lthigh_diff[0])
        quaternion_difference.append(scalar_lthigh)
        # print("scaler",scalar_lthigh)
        ##### lknee
        lknee_euler = self.robot_skeleton.q[9]
        lknee_mocap = self.WalkPositions[self.count, 9]
        quat_lknee = euler2quat(z=0., y=0., x=lknee_euler)
        quat_lknee_mocap = euler2quat(z=0., y=0., x=lknee_mocap)
        lknee_diff = mult(inverse(quat_lknee_mocap), quat_lknee)
        scalar_lknee = 2 * np.arccos(lknee_diff[0])
        quaternion_difference.append(scalar_lknee)
        #### lfoot
        lfoot_euler = self.robot_skeleton.q[10:12]
        lfoot_mocap = self.WalkPositions[self.count, 10:12]
        quat_lfoot = euler2quat(z=lfoot_euler[1], y=0., x=lfoot_euler[0])
        quat_lfoot_mocap = euler2quat(z=lfoot_mocap[1], y=0., x=lfoot_mocap[0])
        lfoot_diff = mult(inverse(quat_lfoot_mocap), quat_lfoot)
        scalar_lfoot = 2 * np.arccos(lfoot_diff[0])
        quaternion_difference.append(scalar_lfoot)
        #### rthigh
        rthigh_euler = self.robot_skeleton.q[12:15]
        rthigh_mocap = self.WalkPositions[self.count, 12:15]
        quat_rthigh = euler2quat(z=rthigh_euler[2], y=rthigh_euler[1], x=rthigh_euler[0])
        quat_rthigh_mocap = euler2quat(z=rthigh_mocap[2], y=rthigh_mocap[1], x=rthigh_mocap[0])
        rthigh_diff = mult(inverse(quat_rthigh_mocap), quat_rthigh)
        scalar_rthigh = 2 * np.arccos(rthigh_diff[0])
        quaternion_difference.append(scalar_rthigh)
        # print("scaler",scalar_lthigh)
        ##### rknee
        rknee_euler = self.robot_skeleton.q[15]
        rknee_mocap = self.WalkPositions[self.count, 15]
        quat_rknee = euler2quat(z=0., y=0., x=rknee_euler)
        quat_rknee_mocap = euler2quat(z=0., y=0., x=rknee_mocap)
        rknee_diff = mult(inverse(quat_rknee_mocap), quat_rknee)
        scalar_rknee = 2 * np.arccos(rknee_diff[0])
        quaternion_difference.append(scalar_rknee)
        #### rfoot
        rfoot_euler = self.robot_skeleton.q[16:18]
        rfoot_mocap = self.WalkPositions[self.count, 16:18]
        quat_rfoot = euler2quat(z=rfoot_euler[1], y=0., x=rfoot_euler[0])
        quat_rfoot_mocap = euler2quat(z=rfoot_mocap[1], y=0., x=rfoot_mocap[0])
        rfoot_diff = mult(inverse(quat_rfoot_mocap), quat_rfoot)
        scalar_rfoot = 2 * np.arccos(rfoot_diff[0])
        quaternion_difference.append(scalar_rfoot)

        scalar_thoraxx = self.robot_skeleton.q[18] - self.WalkPositions[self.count, 18]
        quaternion_difference.append(scalar_thoraxx)
        scalar_thoraxy = self.robot_skeleton.q[19] - self.WalkPositions[self.count, 19]
        quaternion_difference.append(scalar_thoraxy)
        scalar_thoraxz = self.robot_skeleton.q[20] - self.WalkPositions[self.count, 20]
        quaternion_difference.append(scalar_thoraxz)
        #### l upper arm
        larm_euler = self.robot_skeleton.q[21:24]
        larm_mocap = self.WalkPositions[self.count, 21:24]
        quat_larm = euler2quat(z=larm_euler[2], y=larm_euler[1], x=larm_euler[0])
        quat_larm_mocap = euler2quat(z=larm_mocap[2], y=larm_mocap[1], x=larm_mocap[0])
        larm_diff = mult(inverse(quat_larm_mocap), quat_larm)
        scalar_larm = 2 * np.arccos(larm_diff[0])
        quaternion_difference.append(scalar_larm)
        # print("scaler",scalar_lthigh)
        ##### l elbow
        lelbow_euler = self.robot_skeleton.q[24]
        lelbow_mocap = self.WalkPositions[self.count, 24]
        quat_lelbow = euler2quat(z=0., y=0., x=lelbow_euler)
        quat_lelbow_mocap = euler2quat(z=0., y=0., x=lelbow_mocap)
        lelbow_diff = mult(inverse(quat_lelbow_mocap), quat_lelbow)
        scalar_lelbow = 2 * np.arccos(lelbow_diff[0])
        quaternion_difference.append(scalar_lelbow)
        #### r upper arm
        rarm_euler = self.robot_skeleton.q[25:28]
        rarm_mocap = self.WalkPositions[self.count, 25:28]
        quat_rarm = euler2quat(z=rarm_euler[2], y=rarm_euler[1], x=rarm_euler[0])
        quat_rarm_mocap = euler2quat(z=rarm_mocap[2], y=rarm_mocap[1], x=rarm_mocap[0])
        rarm_diff = mult(inverse(quat_rarm_mocap), quat_rarm)
        scalar_rarm = 2 * np.arccos(rarm_diff[0])
        quaternion_difference.append(scalar_rarm)
        # print("scaler",scalar_lthigh)
        ##### r elbow
        relbow_euler = self.robot_skeleton.q[28]
        relbow_mocap = self.WalkPositions[self.count, 28]
        quat_relbow = euler2quat(z=0., y=0., x=relbow_euler)
        quat_relbow_mocap = euler2quat(z=0., y=0., x=relbow_mocap)
        relbow_diff = mult(inverse(quat_relbow_mocap), quat_relbow)
        scalar_relbow = 2 * np.arccos(relbow_diff[0])
        quaternion_difference.append(scalar_relbow)

        quat_reward = np.exp(-2 * np.sum(np.square(quaternion_difference)))

        return quat_reward

