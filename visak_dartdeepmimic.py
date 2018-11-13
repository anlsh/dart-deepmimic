from dartdeepmimic import DartDeepMimicEnv
from dartdeepmimic import JointType
import numpy as np
from euclideanSpace import euler2quat, angle_axis2euler
from quaternions import mult, inverse
from numpy.linalg import norm
import copy
import random
import os
from gym.envs.dart import dart_env

class VisakDartDeepMimicEnv(DartDeepMimicEnv):

    def __init__(self, mocap_vel_path,
                 *args, **kwargs):

        DartDeepMimicEnv.__init__(self, *args, **kwargs)


        #################################################
        # DART INITALIZATION STUFF #
        ############################

        with open(mocap_vel_path,"rb") as fp:
            self.RefDQs = np.loadtxt(fp)

        self.robot_skeleton = self.dart_world.skeletons[1]

        self.robot_skeleton.set_self_collision_check(True)

        for i in range(self.robot_skeleton.njoints-1):
            self.robot_skeleton.joint(i).set_position_limit_enforced(True)
            self.robot_skeleton.dof(i).set_damping_coefficient(10.)

        for body in self.robot_skeleton.bodynodes \
            + self.dart_world.skeletons[0].bodynodes:
           body.set_friction_coeff(20.)

        for jt in range(0, len(self.robot_skeleton.joints)):
            if self.robot_skeleton.joints[jt].has_position_limit(0):
                self.robot_skeleton.joints[jt].set_position_limit_enforced(True)

        #################################################

    def type_lambda(self, joint_name):
        if joint_name == "root1":
            return JointType.TRANS
        else:
            return JointType.ROT

    def _get_ee_positions(self, skel):

        point_rarm = [0.,-0.60,-0.15]
        point_larm = [0.,-0.60,-0.15]
        point_rfoot = [0.,0.,-0.20]
        point_lfoot = [0.,0.,-0.20]

        global_rarm = skel.bodynodes[16].to_world(point_rarm)
        global_larm = skel.bodynodes[13].to_world(point_larm)
        global_lfoot = skel.bodynodes[4].to_world(point_lfoot)
        global_rfoot = skel.bodynodes[7].to_world(point_rfoot)

        return np.array([global_rarm, global_rarm,
                         global_rfoot, global_lfoot])

    def ClampTorques(self,torques):
        torqueLimits = np.array([150.0*5,
                                 80.*3,
                                 80.*3,
                                 100.*5,
                                 80.*5,
                                 60.,
                                 150.0*5,
                                 80.*3,
                                 80.*3,
                                 100.*5,
                                 80.*5,
                                 60.,
                                 150.*5,
                                 150.*5,
                                 150.*5,
                                 10.,
                                 5.,
                                 5.,
                                 5.,
                                 10.,
                                 5.,
                                 5,
                                 5.])*2

        for i in range(6,self.robot_skeleton.ndofs):
            if torques[i] > torqueLimits[i-6]:
                torques[i] = torqueLimits[i-6]
            if torques[i] < -torqueLimits[i-6]:
                torques[i] = -torqueLimits[i-6]

        return torques

    def PID(self, skel, target):
        self.kp = np.array([250]*23)
        self.kd = np.array([0.005]*23)

        self.kp[0] = 600+25
        self.kp[3] = 225+25
        self.kp[9] = 225+25
        self.kp[10] = 200
        self.kp[16] = 200
        self.kp[[1,2]] = 150
        self.kp[[7,8]] = 150
        self.kp[6] = 600+25
        self.kp[15:] = 155
        self.kd[15:]= 0.05

        self.kp = [item/2 for item in self.kp]
        self.kd = [item/2 for item in self.kd]

        q = skel.q
        qdot = skel.dq
        tau = np.zeros((self.robot_skeleton.ndofs,))
        for i in range(6, self.robot_skeleton.ndofs):
            tau[i] = -self.kp[i - 6] * \
                (q[i] - target[i]) - \
                self.kd[i - 6] *qdot[i]

        torqs = self.ClampTorques(tau)

        return torqs[6:]

    def vel_diff(self, skel, framenum):

        Joint_weights = np.ones(23,)
        Joint_weights[[0,3,6,9,16,20,10,16]] = 10
        Weight_matrix = np.diag(Joint_weights)

        vel_diff = self.RefDQs[framenum,6:] - skel.dq[6:]

        return np.sum(vel_diff.T*Weight_matrix*vel_diff)

    def should_terminate(self):

        skel = self.robot_skeleton
        obs = self.state_vector()

        height = skel.bodynodes[0].com()[1]

        return not (np.isfinite(obs).all()
                    and (np.abs(obs[2:]) < 200).all()
                    and (height > -0.70) and (height < 0.40)
                    and (abs(skel.q[4]) < 0.30)
                    and (abs(skel.q[5]) < 0.50)
                    and (skel.q[3] > -0.4)
                    and (skel.q[3] < 0.3))

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = 0
            self._get_viewer().scene.tb.trans[2] = -7.5
            self._get_viewer().scene.tb.trans[1] = 0
            #-10.0

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer().close()
                self.viewer = None
            return

        if mode == 'rgb_array':
            data = self._get_viewer().getFrame()
            return data
        elif mode == 'human':
            self._get_viewer().runSingleStep()

