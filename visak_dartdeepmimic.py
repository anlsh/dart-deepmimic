from dartdeepmimic import DartDeepMimicEnv, map_dofs, END_OFFSET
import ddm_argparse
import numpy as np
import argparse
import runner
from copy import copy
from euclideanSpace import euler2quat, angle_axis2euler
from quaternions import mult, inverse


def transformActions(actions):
    joint_targets = np.zeros(23, )
        # Left thigh
    lthigh = actions[:4]
    euler_lthigh = angle_axis2euler(theta=lthigh[0], vector=lthigh[1:])
    joint_targets[0] = euler_lthigh[2]
    joint_targets[1] = euler_lthigh[1]
    joint_targets[2] = euler_lthigh[0]
    ###### Left Knee
    joint_targets[3] = actions[4]
    ### left foot
    lfoot = actions[5:9]
    euler_lfoot = angle_axis2euler(theta=lfoot[0], vector=lfoot[1:])
    joint_targets[4] = euler_lfoot[2]
    joint_targets[5] = euler_lfoot[0]

    # right thigh
    rthigh = actions[9:13]
    euler_rthigh = angle_axis2euler(theta=rthigh[0], vector=rthigh[1:])
    joint_targets[6] = euler_rthigh[2]
    joint_targets[7] = euler_rthigh[1]
    joint_targets[8] = euler_rthigh[0]
    ###### right Knee
    joint_targets[9] = actions[13]
    ### right foot
    rfoot = actions[14:18]
    euler_rfoot = angle_axis2euler(theta=rfoot[0], vector=rfoot[1:])
    joint_targets[10] = euler_rfoot[2]
    joint_targets[11] = euler_rfoot[0]

    ###thorax

    thorax = actions[18:22]
    euler_thorax = angle_axis2euler(theta=thorax[0], vector=thorax[1:])
    joint_targets[12] = euler_thorax[2]
    joint_targets[13] = euler_thorax[1]
    joint_targets[14] = euler_thorax[0]

    #### l upper arm
    l_arm = actions[22:26]
    euler_larm = angle_axis2euler(theta=l_arm[0], vector=l_arm[1:])
    joint_targets[15] = euler_larm[2]
    joint_targets[16] = euler_larm[1]
    joint_targets[17] = euler_larm[0]

    ## l elbow

    joint_targets[18] = actions[25]

    ## r upper arm
    r_arm = actions[27:31]
    euler_rarm = angle_axis2euler(theta=r_arm[0], vector=r_arm[1:])
    joint_targets[19] = euler_rarm[2]
    joint_targets[20] = euler_rarm[1]
    joint_targets[21] = euler_rarm[0]

    ###r elbow

    joint_targets[22] = actions[30]
    return joint_targets

class VisakDartDeepMimicEnv(DartDeepMimicEnv):

    def __init__(self, *args, **kwargs):


        self.WalkPositions = None
        self.WalkVelocities = None

        super(VisakDartDeepMimicEnv, self).__init__(*args, **kwargs)

    # def q_from_netvector(self, netvector):

    #     myaction = super(VisakDartDeepMimicEnv,
    #                      self).q_from_netvector(netvector)

    #     vsk_action = transformActions(netvector)
    #     print("actiondiff: ", np.subtract(myaction[6:], vsk_action))

    #     return myaction


    def construct_frames(self, ref_motion_path):
        """
        AMC data is given in sequential degrees, while dart specifies angles
        in rotating radians. The conversion is quite expensive, so we precomute
        all positions and velocities and store the results
        """

        raw_framelist = None

        with open("assets/mocap/walk/WalkPositions_corrected.txt","rb") as fp:
            self.WalkPositions = np.loadtxt(fp)
        with open("assets/mocap/walk/WalkPositions_corrected.txt","rb") as fp:
            self.WalkVelocities = np.loadtxt(fp)
        with open("assets/mocap/walk/rarm_endeffector.txt", "rb") as fp:
            self.rarm_endeffector = np.loadtxt(fp)
        with open("assets/mocap/walk/larm_endeffector.txt", "rb") as fp:
            self.larm_endeffector = np.loadtxt(fp)
        with open("assets/mocap/walk/lfoot_endeffector.txt", "rb") as fp:
            self.lfoot_endeffector = np.loadtxt(fp)
        with open("assets/mocap/walk/rfoot_endeffector.txt", 'rb') as fp:
            self.rfoot_endeffector = np.loadtxt(fp)
        with open("assets/mocap/walk/com.txt", 'rb') as fp:
            self.com = np.loadtxt(fp)

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
            updated_vel[3:6] = updated_vel[0:3][::-1]
            updated_vel[0:3] = temp

            map_dofs(self.ref_skel.dofs, updated_pos,
                     updated_vel, 0, 0)
            pos_frames[i] = updated_pos
            vel_frames[i] = updated_vel
            # com_frames[i] = self.ref_skel.com()
            # com_frames[i] = self.com[i][::-1]
            # quat_frames[i] = self.quaternion_angles(self.ref_skel)
            # TODO Parse actual end positions
            # ee_frames[i] = [self.ref_skel.bodynodes[ii].to_world(END_OFFSET)
            #                 for ii in self._end_effector_indices]

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

    def reward(self, skel, framenum):

        return self.vsk_reward(skel, framenum)

    def vsk_reward(self, skel, framenum):

        point_rarm = [0., -0.60, -0.15]
        point_larm = [0., -0.60, -0.15]
        point_rfoot = [0., 0., -0.20]
        point_lfoot = [0., 0., -0.20]

        global_rarm = skel.bodynodes[16].to_world(point_rarm)
        global_larm = skel.bodynodes[13].to_world(point_larm)
        global_lfoot = skel.bodynodes[4].to_world(point_lfoot)
        global_rfoot = skel.bodynodes[7].to_world(point_rfoot)

        height = skel.bodynodes[0].com()[1]

        Joint_weights = np.ones(23, )
        Joint_weights[[0, 3, 6, 9, 16, 20, 10, 16]] = 10

        Weight_matrix = np.diag(Joint_weights)

        rarm_term = np.sum(np.square(self.rarm_endeffector[framenum, :] - global_rarm))
        larm_term = np.sum(np.square(self.larm_endeffector[framenum, :] - global_larm))
        rfoot_term = np.sum(np.square(self.rfoot_endeffector[framenum, :] - global_rfoot))
        lfoot_term = np.sum(np.square(self.lfoot_endeffector[framenum, :] - global_lfoot))

        end_effector_reward = np.exp(-40 * (rarm_term + larm_term + rfoot_term + lfoot_term))
        com_reward = np.exp(-40 * np.sum(np.square(self.com[framenum, :] - skel.bodynodes[0].com())))

        vel_diff = self.WalkVelocities[framenum, 6:] - skel.dq[6:]
        vel_pen = np.sum(vel_diff.T * Weight_matrix * vel_diff)
        joint_vel_term = 1 * np.asarray(np.exp(-1e-1 * vel_pen))

        quat_term = self.vsk_quatreward()
        reward = 0.1 * end_effector_reward + 0.1 * joint_vel_term + 0.25 * com_reward + 1.65 * quat_term
        return reward


    def vsk_quatreward(self, ):
        quaternion_difference = []

        #### lthigh
        lthigh_euler = self.control_skel.q[6:9]
        lthigh_mocap = self.WalkPositions[self.framenum, 6:9]
        quat_lthigh = euler2quat(z=lthigh_euler[2], y=lthigh_euler[1], x=lthigh_euler[0])
        quat_lthigh_mocap = euler2quat(z=lthigh_mocap[2], y=lthigh_mocap[1], x=lthigh_mocap[0])
        lthigh_diff = mult(inverse(quat_lthigh_mocap), quat_lthigh)
        scalar_lthigh = 2 * np.arccos(lthigh_diff[0])
        quaternion_difference.append(scalar_lthigh)

        ##### lknee
        lknee_euler = self.control_skel.q[9]
        lknee_mocap = self.WalkPositions[self.framenum, 9]
        quat_lknee = euler2quat(z=0., y=0., x=lknee_euler)
        quat_lknee_mocap = euler2quat(z=0., y=0., x=lknee_mocap)
        lknee_diff = mult(inverse(quat_lknee_mocap), quat_lknee)
        scalar_lknee = 2 * np.arccos(lknee_diff[0])
        quaternion_difference.append(scalar_lknee)

        #### lfoot
        lfoot_euler = self.control_skel.q[10:12]
        lfoot_mocap = self.WalkPositions[self.framenum, 10:12]
        quat_lfoot = euler2quat(z=lfoot_euler[1], y=0., x=lfoot_euler[0])
        quat_lfoot_mocap = euler2quat(z=lfoot_mocap[1], y=0., x=lfoot_mocap[0])
        lfoot_diff = mult(inverse(quat_lfoot_mocap), quat_lfoot)
        scalar_lfoot = 2 * np.arccos(lfoot_diff[0])
        quaternion_difference.append(scalar_lfoot)

        #### rthigh
        rthigh_euler = self.control_skel.q[12:15]
        rthigh_mocap = self.WalkPositions[self.framenum, 12:15]
        quat_rthigh = euler2quat(z=rthigh_euler[2], y=rthigh_euler[1], x=rthigh_euler[0])
        quat_rthigh_mocap = euler2quat(z=rthigh_mocap[2], y=rthigh_mocap[1], x=rthigh_mocap[0])
        rthigh_diff = mult(inverse(quat_rthigh_mocap), quat_rthigh)
        scalar_rthigh = 2 * np.arccos(rthigh_diff[0])
        quaternion_difference.append(scalar_rthigh)

        ##### rknee
        rknee_euler = self.control_skel.q[15]
        rknee_mocap = self.WalkPositions[self.framenum, 15]
        quat_rknee = euler2quat(z=0., y=0., x=rknee_euler)
        quat_rknee_mocap = euler2quat(z=0., y=0., x=rknee_mocap)
        rknee_diff = mult(inverse(quat_rknee_mocap), quat_rknee)
        scalar_rknee = 2 * np.arccos(rknee_diff[0])
        quaternion_difference.append(scalar_rknee)

        #### rfoot
        rfoot_euler = self.control_skel.q[16:18]
        rfoot_mocap = self.WalkPositions[self.framenum, 16:18]
        quat_rfoot = euler2quat(z=rfoot_euler[1], y=0., x=rfoot_euler[0])
        quat_rfoot_mocap = euler2quat(z=rfoot_mocap[1], y=0., x=rfoot_mocap[0])
        rfoot_diff = mult(inverse(quat_rfoot_mocap), quat_rfoot)
        scalar_rfoot = 2 * np.arccos(rfoot_diff[0])
        quaternion_difference.append(scalar_rfoot)

        # Abdoemn/thorax

        ###############################################
        # ALERT ALERT CHANGING THE GOOD AND HOLY CODE #
        ###############################################

        # scalar_thoraxx = self.control_skel.q[18] - self.WalkPositions[self.framenum, 18]
        # quaternion_difference.append(scalar_thoraxx)
        # scalar_thoraxy = self.control_skel.q[19] - self.WalkPositions[self.framenum, 19]
        # quaternion_difference.append(scalar_thoraxy)
        # scalar_thoraxz = self.control_skel.q[20] - self.WalkPositions[self.framenum, 20]
        # quaternion_difference.append(scalar_thoraxz)

        ##################################################
        # THE DEFILED CODE (although the above is weird) #
        ##################################################
        thorax_euler  = self.control_skel.q[18:21]
        thorax_mocap  = self.WalkPositions[self.framenum, 18:21]
        quat_thorax = euler2quat(*thorax_euler)
        quat_thorax_mc = euler2quat(*thorax_mocap)
        thing_diff = mult(inverse(quat_thorax_mc), quat_thorax)
        scalar_thorax = 2 * np.arccos(thing_diff[0])
        quaternion_difference.append(scalar_thorax)

        #### l upper arm
        larm_euler = self.control_skel.q[21:24]
        larm_mocap = self.WalkPositions[self.framenum, 21:24]
        quat_larm = euler2quat(z=larm_euler[2], y=larm_euler[1], x=larm_euler[0])
        quat_larm_mocap = euler2quat(z=larm_mocap[2], y=larm_mocap[1], x=larm_mocap[0])
        larm_diff = mult(inverse(quat_larm_mocap), quat_larm)
        scalar_larm = 2 * np.arccos(larm_diff[0])
        quaternion_difference.append(scalar_larm)

        ##### l elbow
        lelbow_euler = self.control_skel.q[24]
        lelbow_mocap = self.WalkPositions[self.framenum, 24]
        quat_lelbow = euler2quat(z=0., y=0., x=lelbow_euler)
        quat_lelbow_mocap = euler2quat(z=0., y=0., x=lelbow_mocap)
        lelbow_diff = mult(inverse(quat_lelbow_mocap), quat_lelbow)
        scalar_lelbow = 2 * np.arccos(lelbow_diff[0])
        quaternion_difference.append(scalar_lelbow)

        #### r upper arm
        rarm_euler = self.control_skel.q[25:28]
        rarm_mocap = self.WalkPositions[self.framenum, 25:28]
        quat_rarm = euler2quat(z=rarm_euler[2], y=rarm_euler[1], x=rarm_euler[0])
        quat_rarm_mocap = euler2quat(z=rarm_mocap[2], y=rarm_mocap[1], x=rarm_mocap[0])
        rarm_diff = mult(inverse(quat_rarm_mocap), quat_rarm)
        scalar_rarm = 2 * np.arccos(rarm_diff[0])
        quaternion_difference.append(scalar_rarm)

        ##### r elbow
        relbow_euler = self.control_skel.q[28]
        relbow_mocap = self.WalkPositions[self.framenum, 28]
        quat_relbow = euler2quat(z=0., y=0., x=relbow_euler)
        quat_relbow_mocap = euler2quat(z=0., y=0., x=relbow_mocap)
        relbow_diff = mult(inverse(quat_relbow_mocap), quat_relbow)
        scalar_relbow = 2 * np.arccos(relbow_diff[0])
        quaternion_difference.append(scalar_relbow)

        quat_reward = np.exp(-2 * np.sum(np.square(quaternion_difference)))

        return quat_reward

