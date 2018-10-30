from dartdeepmimic import DartDeepMimicEnv
import numpy as np
from euclideanSpace import euler2quat, angle_axis2euler
from quaternions import mult, inverse
from numpy.linalg import norm
import copy

class VisakDartDeepMimicEnv(DartDeepMimicEnv):

    def __init__(self, *args, **kwargs):

        super(VisakDartDeepMimicEnv, self).__init__(*args, **kwargs)

        # TODO The following stuff is probably... wrong
        # However it must be done in order to bring us into P A R I T Y

        self.robot_skeleton.set_self_collision_check(True)

        for i in range(self.robot_skeleton.njoints-1):
            self.robot_skeleton.joint(i).set_position_limit_enforced(True)
            self.robot_skeleton.dof(i).set_damping_coefficient(10.)

        for body in self.robot_skeleton.bodynodes+self.dart_world.skeletons[0].bodynodes:
           body.set_friction_coeff(20.)

        for jt in range(0, len(self.robot_skeleton.joints)):
            if self.robot_skeleton.joints[jt].has_position_limit(0):
                self.robot_skeleton.joints[jt].set_position_limit_enforced(True)

        # END UNHOLY CODE

    def construct_frames(self, ref_skel, ref_motion_path):

        with open("assets/mocap/walk/WalkPositions_corrected.txt",
                  "rb") as fp:
            ref_q_frames = np.loadtxt(fp)
        with open("assets/mocap/walk/WalkVelocities_corrected.txt",
                  "rb") as fp:
            MotionVelocities = np.loadtxt(fp)

        ####################################################
        # TODO Useless except to override my mocap parsing #

        prefix = "assets/mocap/walk/"

        with open(prefix+"rarm_endeffector.txt","rb") as fp:
            rarm_endeffector = np.loadtxt(fp)[:-1]

        with open(prefix+"larm_endeffector.txt","rb") as fp:
            larm_endeffector = np.loadtxt(fp)[:-1]

        with open(prefix+"lfoot_endeffector.txt","rb") as fp:
            lfoot_endeffector = np.loadtxt(fp)[:-1]

        with open(prefix+"rfoot_endeffector.txt",'rb') as fp:
            rfoot_endeffector = np.loadtxt(fp)[:-1]

        with open(prefix+"com.txt",'rb') as fp:
            com = np.loadtxt(fp)[:-1]

        ####################################################

        num_frames = len(ref_q_frames)

        pos_frames = [None] * num_frames
        vel_frames = [None] * num_frames
        quat_frames = [None] * num_frames
        com_frames = [None] * num_frames
        ee_frames = [None] * num_frames

        for i in range(num_frames):

            updated_pos = ref_q_frames[i,:]
            updated_vel = MotionVelocities[i,:]

            pos_frames[i] = updated_pos.copy()
            vel_frames[i] = updated_vel.copy()

            ref_skel.set_positions(pos_frames[i])
            ref_skel.set_velocities(vel_frames[i])

            ###########################################################
            # TODO Re-enable my own code parsing
            # Code conflicts: For some reason, the com / ee frames Visak and I
            # parse are different, therefore I discard my version of it

            # MY VERSION BELOW
            # com_frames[i] = ref_skel.bodynodes[0].com()
            # ee_frames[i] = self._get_ee_positions(ref_skel)

            com_frames[i] = com[i]
            ee_frames[i] = [rarm_endeffector[i], larm_endeffector[i],
                            rfoot_endeffector[i], lfoot_endeffector[i]]
            ##########################################################

        return np.array(pos_frames), np.array(vel_frames), \
            np.array(quat_frames), np.array(com_frames), \
            np.array(ee_frames)

    def _get_ee_positions(self, skel):

        # DIFF Done *Exactly* as visak calculates them

        point_rarm = [0., -0.60, -0.15]
        point_larm = [0., -0.60, -0.15]
        point_rfoot = [0., 0., -0.20]
        point_lfoot = [0., 0., -0.20]

        return np.array([skel.bodynodes[16].to_world(point_rarm),
                         skel.bodynodes[13].to_world(point_larm),
                         skel.bodynodes[7].to_world(point_rfoot),
                         skel.bodynodes[4].to_world(point_lfoot)])

    def should_terminate(self, newstate):
        """
        Returns a tuple of (done, rude_termination)

        If rude termination is given as true, zero reward will be
        yielded from the state
        """
        term, rude_term = False, False

        if self.framenum >= self.num_frames - 1:
            term, rude_term = True, False
        if not ((np.abs(newstate[2:]) < 200).all()
                and (self.robot_skeleton.bodynodes[0].com()[1] > -0.7)
                and (self.robot_skeleton.q[3] > -0.4)
                and (self.robot_skeleton.q[3] < 0.3)
                and (abs(self.robot_skeleton.q[4]) < 0.30)
                and (abs(self.robot_skeleton.q[5]) < 0.30)):
            term, rude_term = True, True

        if rude_term and not term:
            raise RuntimeError("Must terminate to rudely terminate")
        return term, rude_term

    def PID(self, skel, actuated_angle_targets):

        kp = np.array([250.] * 23)
        kd = np.array([0.005] * 23)

        kp[0] = 600 + 25
        kp[3] = 225 + 25
        kp[9] = 225 + 25
        kp[10] = 200
        kp[16] = 200
        kp[[1, 2]] = 150
        kp[[7, 8]] = 150
        kp[6] = 600 + 25
        kp[15:] = 155

        kd[15:] = 0.05

        kp = np.multiply(.5, kp)
        kd = np.multiply(.5, kd)

        tau_p = np.multiply(kp, actuated_angle_targets - skel.q[6:])
        tau_d = np.multiply(kd, skel.dq[6:])

        tau = tau_p - tau_d

        # DIFF I use a more elegant method of clipping, but is equivalent
        TORQUE_LIMITS = np.array([150.0 * 5,
                                  80. * 3,
                                  80. * 3,
                                  100. * 5,
                                  80. * 5,
                                  60.,
                                  150.0 * 5,
                                  80. * 3,
                                  80. * 3,
                                  100. * 5,
                                  80. * 5,
                                  60.,
                                  150. * 5,
                                  150. * 5,
                                  150. * 5,
                                  10.,
                                  5.,
                                  5.,
                                  5.,
                                  10.,
                                  5.,
                                  5,
                                  5.]) * 2

        return np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

    def targets_from_netvector(self, actions):

        joint_targets = np.zeros(23)

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

        joint_targets[18] = actions[26]

        ## r upper arm
        r_arm = actions[27:31]
        euler_rarm = angle_axis2euler(theta=r_arm[0], vector=r_arm[1:])
        joint_targets[19] = euler_rarm[2]
        joint_targets[20] = euler_rarm[1]
        joint_targets[21] = euler_rarm[0]

        ###r elbow

        joint_targets[22] = actions[31]

        return joint_targets

    def ee_reward(self, skel, framenum):
        ee_positions = self._get_ee_positions(skel)
        ref_ee_positions = self.ref_ee_frames[framenum]
        ee_diffs = ee_positions - ref_ee_positions
        ee_diffmag = np.sum([np.sum(np.square(diff)) for diff in ee_diffs])
        return np.exp(-40 * ee_diffmag)

    def com_reward(self, skel, framenum):
        return np.exp(-40 * np.sum(np.square(self.ref_com_frames[framenum]
                                             - skel.bodynodes[0].com())))

    def vel_reward(self, skel, framenum):
        Joint_weights = np.ones(23, )
        Joint_weights[[0, 3, 6, 9, 16, 20, 10, 16]] = 10
        Weight_matrix = np.diag(Joint_weights)

        vel_diff = self.ref_dq_frames[framenum][6:] - skel.dq[6:]
        vel_pen = np.sum(vel_diff.T * Weight_matrix * vel_diff)
        return np.exp(-0.1 * vel_pen)

    def reward(self, skel, framenum):

        R_ee = self.ee_reward(skel, framenum)
        R_com = self.com_reward(skel, framenum)
        R_vel = self.vel_reward(skel, framenum)
        R_quat = self.quat_reward(skel, framenum)

        reward = 0.1 * R_ee + 0.1 * R_vel \
                 + 0.15 * R_com + .65 * R_quat

        return reward

    def quat_reward(self, skel, framenum):

        # DIFF Quaternion difference (and therefore reward) are computed
        # same as Visak does, but for the fact that I do some finiteness
        # checks before returning anything

        quaternion_difference = []

        #### lthigh
        lthigh_euler = skel.q[6:9]
        lthigh_mocap = self.ref_q_frames[framenum][6:9]
        quat_lthigh = euler2quat(z=lthigh_euler[2], y=lthigh_euler[1], x=lthigh_euler[0])
        quat_lthigh_mocap = euler2quat(z=lthigh_mocap[2], y=lthigh_mocap[1], x=lthigh_mocap[0])
        lthigh_diff = mult(inverse(quat_lthigh_mocap), quat_lthigh)
        scalar_lthigh = 2 * np.arccos(lthigh_diff[0])
        quaternion_difference.append(scalar_lthigh)

        ##### lknee
        lknee_euler = skel.q[9]
        lknee_mocap = self.ref_q_frames[framenum][9]
        quat_lknee = euler2quat(z=0., y=0., x=lknee_euler)
        quat_lknee_mocap = euler2quat(z=0., y=0., x=lknee_mocap)
        lknee_diff = mult(inverse(quat_lknee_mocap), quat_lknee)
        scalar_lknee = 2 * np.arccos(lknee_diff[0])
        quaternion_difference.append(scalar_lknee)

        #### lfoot
        lfoot_euler = skel.q[10:12]
        lfoot_mocap = self.ref_q_frames[framenum][10:12]
        quat_lfoot = euler2quat(z=lfoot_euler[1], y=0., x=lfoot_euler[0])
        quat_lfoot_mocap = euler2quat(z=lfoot_mocap[1], y=0., x=lfoot_mocap[0])
        lfoot_diff = mult(inverse(quat_lfoot_mocap), quat_lfoot)
        scalar_lfoot = 2 * np.arccos(lfoot_diff[0])
        quaternion_difference.append(scalar_lfoot)

        #### rthigh
        rthigh_euler = skel.q[12:15]
        rthigh_mocap = self.ref_q_frames[framenum][12:15]
        quat_rthigh = euler2quat(z=rthigh_euler[2], y=rthigh_euler[1], x=rthigh_euler[0])
        quat_rthigh_mocap = euler2quat(z=rthigh_mocap[2], y=rthigh_mocap[1], x=rthigh_mocap[0])
        rthigh_diff = mult(inverse(quat_rthigh_mocap), quat_rthigh)
        scalar_rthigh = 2 * np.arccos(rthigh_diff[0])
        quaternion_difference.append(scalar_rthigh)

        ##### rknee
        rknee_euler = skel.q[15]
        rknee_mocap = self.ref_q_frames[framenum][15]
        quat_rknee = euler2quat(z=0., y=0., x=rknee_euler)
        quat_rknee_mocap = euler2quat(z=0., y=0., x=rknee_mocap)
        rknee_diff = mult(inverse(quat_rknee_mocap), quat_rknee)
        scalar_rknee = 2 * np.arccos(rknee_diff[0])
        quaternion_difference.append(scalar_rknee)

        #### rfoot
        rfoot_euler = skel.q[16:18]
        rfoot_mocap = self.ref_q_frames[framenum][16:18]
        quat_rfoot = euler2quat(z=rfoot_euler[1], y=0., x=rfoot_euler[0])
        quat_rfoot_mocap = euler2quat(z=rfoot_mocap[1], y=0., x=rfoot_mocap[0])
        rfoot_diff = mult(inverse(quat_rfoot_mocap), quat_rfoot)
        scalar_rfoot = 2 * np.arccos(rfoot_diff[0])
        quaternion_difference.append(scalar_rfoot)

        # Abdoemn/thorax

        scalar_thoraxx = skel.q[18] - self.ref_q_frames[framenum][18]
        quaternion_difference.append(scalar_thoraxx)
        scalar_thoraxy = skel.q[19] - self.ref_q_frames[framenum][19]
        quaternion_difference.append(scalar_thoraxy)
        scalar_thoraxz = skel.q[20] - self.ref_q_frames[framenum][20]
        quaternion_difference.append(scalar_thoraxz)

        #### l upper arm
        larm_euler = skel.q[21:24]
        larm_mocap = self.ref_q_frames[framenum][21:24]
        quat_larm = euler2quat(z=larm_euler[2], y=larm_euler[1], x=larm_euler[0])
        quat_larm_mocap = euler2quat(z=larm_mocap[2], y=larm_mocap[1], x=larm_mocap[0])
        larm_diff = mult(inverse(quat_larm_mocap), quat_larm)
        scalar_larm = 2 * np.arccos(larm_diff[0])
        quaternion_difference.append(scalar_larm)

        ##### l elbow
        lelbow_euler = skel.q[24]
        lelbow_mocap = self.ref_q_frames[framenum][24]
        quat_lelbow = euler2quat(z=0., y=0., x=lelbow_euler)
        quat_lelbow_mocap = euler2quat(z=0., y=0., x=lelbow_mocap)
        lelbow_diff = mult(inverse(quat_lelbow_mocap), quat_lelbow)
        scalar_lelbow = 2 * np.arccos(lelbow_diff[0])
        quaternion_difference.append(scalar_lelbow)

        #### r upper arm
        rarm_euler = skel.q[25:28]
        rarm_mocap = self.ref_q_frames[framenum][25:28]
        quat_rarm = euler2quat(z=rarm_euler[2], y=rarm_euler[1], x=rarm_euler[0])
        quat_rarm_mocap = euler2quat(z=rarm_mocap[2], y=rarm_mocap[1], x=rarm_mocap[0])
        rarm_diff = mult(inverse(quat_rarm_mocap), quat_rarm)
        scalar_rarm = 2 * np.arccos(rarm_diff[0])
        quaternion_difference.append(scalar_rarm)

        ##### r elbow
        relbow_euler = skel.q[28]
        relbow_mocap = self.ref_q_frames[framenum][28]
        quat_relbow = euler2quat(z=0., y=0., x=relbow_euler)
        quat_relbow_mocap = euler2quat(z=0., y=0., x=relbow_mocap)
        relbow_diff = mult(inverse(quat_relbow_mocap), quat_relbow)
        scalar_relbow = 2 * np.arccos(relbow_diff[0])
        quaternion_difference.append(scalar_relbow)

        quaternion_difference = np.array(quaternion_difference)
        # TODO Re-enable my finiteness checks!!!!
        # quaternion_difference[np.isinf(quaternion_difference)] = 0
        # quaternion_difference[np.isnan(quaternion_difference)] = 0

        quat_reward = np.exp(-2 * np.sum(np.square(quaternion_difference)))

        return quat_reward

    def _get_obs(self, skel=None):

        if skel is None:
            skel = self.robot_skeleton

        # DIFF Visak doesn't specify all the information I do for the abdomen
        # but is unlikely to explain performance gap

        state = np.array([])

        # observation for left leg thigh######################################
        RelPos_lthigh = skel.bodynodes[2].com() - skel.bodynodes[0].com()
        LinVel_lthigh = skel.bodynodes[2].dC
        quat_lthigh = euler2quat(z=skel.q[8], y=skel.q[7], x=skel.q[6])

        state = np.concatenate((state,
                                RelPos_lthigh,
                                quat_lthigh,
                                LinVel_lthigh,
                                skel.dq[6:9]))
        ################################################################3
        RelPos_lknee = skel.bodynodes[3].com() - skel.bodynodes[0].com()
        LinVel_lknee = skel.bodynodes[3].dC
        quat_lknee = euler2quat(z=0., y=0., x=skel.q[9])

        state = np.concatenate((state,
                                RelPos_lknee,
                                quat_lknee,
                                LinVel_lknee,
                                skel.dq[9: 9 + 1]))
        #######################################################################3
        RelPos_lfoot = skel.bodynodes[4].com() - skel.bodynodes[0].com()
        LinVel_lfoot = skel.bodynodes[4].dC
        quat_lfoot = euler2quat(z=skel.q[11], y=0, x=skel.q[10])

        state = np.concatenate((state,
                            RelPos_lfoot,
                            quat_lfoot,
                            LinVel_lfoot,
                            skel.dq[10:12]))
        #######################################################################3
        RelPos_rthigh = skel.bodynodes[5].com() - skel.bodynodes[0].com()
        LinVel_rthigh = skel.bodynodes[5].dC
        quat_rthigh = euler2quat(z=skel.q[14], y=skel.q[13], x=skel.q[12])

        state = np.concatenate((state,
                            RelPos_rthigh,
                            quat_rthigh,
                            LinVel_rthigh,
                            skel.dq[12:15]))
        #####################################################################
        RelPos_rknee = skel.bodynodes[6].com() - skel.bodynodes[0].com()
        LinVel_rknee = skel.bodynodes[6].dC
        quat_rknee = euler2quat(z=0., y=0., x=skel.q[15])

        state = np.concatenate((state,
                                RelPos_rknee,
                                quat_rknee,
                                LinVel_rknee,
                                skel.dq[15: 15 + 1]))

        #####################################################################
        RelPos_rfoot = skel.bodynodes[7].com() - skel.bodynodes[0].com()
        LinVel_rfoot = skel.bodynodes[7].dC
        quat_rfoot = euler2quat(z=skel.q[17], y=0, x=skel.q[16])

        state = np.concatenate((state,
                                RelPos_rfoot,
                                quat_rfoot,
                                LinVel_rfoot,
                                skel.dq[16:18]))

        ###########################################################
        RelPos_larm = skel.bodynodes[12].com() - skel.bodynodes[0].com()
        LinVel_larm = skel.bodynodes[12].dC
        quat_larm = euler2quat(z=skel.q[23], y=skel.q[22], x=skel.q[21])

        state = np.concatenate((state,
                                RelPos_larm,
                                quat_larm,
                                LinVel_larm,
                                skel.dq[21:24]))
        ##############################################################
        RelPos_lelbow = skel.bodynodes[13].com() - skel.bodynodes[0].com()
        LinVel_lelbow = skel.bodynodes[13].dC
        quat_lelbow = euler2quat(z=0., y=0., x=skel.q[24])

        state = np.concatenate((state,
                                RelPos_lelbow,
                                quat_lelbow,
                                LinVel_lelbow,
                                skel.dq[24:24 + 1]))

        ################################################################
        RelPos_rarm = skel.bodynodes[15].com() - skel.bodynodes[0].com()
        LinVel_rarm = skel.bodynodes[15].dC
        quat_rarm = euler2quat(z=skel.q[27], y=skel.q[26], x=skel.q[25])

        state = np.concatenate((state,
                                RelPos_rarm,
                                quat_rarm,
                                LinVel_rarm,
                                skel.dq[25:28]))
        #################################################################3
        RelPos_relbow = skel.bodynodes[16].com() - skel.bodynodes[0].com()
        LinVel_relbow = skel.bodynodes[16].dC
        quat_relbow = euler2quat(z=0., y=0., x=skel.q[28])

        state = np.concatenate((state,
                                RelPos_relbow,
                                quat_relbow,
                                LinVel_relbow,
                                skel.dq[28: 28 + 1]))

        ############ ABDOMEN ########################

        # TODO CODE CONFLICTS

        ###############################
        # MY CODE WHICH IS CONSISTENT #
        ###############################

        # RelPos_abdomen = skel.bodynodes[7].com() - skel.bodynodes[0].com()
        # LinVel_abdomen = skel.bodynodes[7].dC
        # quat_abdomen = euler2quat(z=skel.q[20], y=skel.q[19], x=skel.q[18])

        # state = np.concatenate((state,
        #                         RelPos_abdomen,
        #                         quat_abdomen,
        #                         LinVel_abdomen,
        #                         skel.dq[18:21]))

        ##################################
        # THE CODE HE USES FOR NO REASON #
        ##################################

        state = np.concatenate((state,
                                skel.q[18:21],
                                skel.dq[18:21]))

        ########################
        # END CONFLICTING CODE #
        ########################

        state = np.concatenate((state, [self.framenum / self.num_frames]))

        return state

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = 0.0
            self._get_viewer().scene.tb.trans[2] = -7.5
            self._get_viewer().scene.tb.trans[1] = 0.0
