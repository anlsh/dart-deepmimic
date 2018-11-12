__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import pickle
from scipy import signal
import numpy.linalg as la
import copy

import baselines.common.tf_util as U
import tensorflow as tf
from baselines.ppo1.mlp_policy import MlpPolicy
from gym import wrappers,spaces
from euclideanSpace import *
from quaternions import *
import os

class DartHumanoid3D_cartesian(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):


        self.control_bounds = np.array([10*np.ones(32,), -10*np.ones(32,)])

        obs_dim = 127#+3#for phase
        ### POINTS ON THE FOOT PLANE
        self.P = np.array([0.,0.,-0.0525])
        self.Q = np.array([-0.05,0.,-0.05])
        self.R = np.array([-0.05,0.,0.])

        self.duplicate = False
        self.switch = -1
        self.impactCount = 0
        self.storeState = False
        self.init_q = np.zeros(29,)
        self.init_dq = np.zeros(29,)
        ### LOGGING
        self.dumpTorques = False
        self.dumpRewards = False
        self.dumpActions = False
        self.dumpCOM = False
        self.dumpStates = False
        #### PRINTING
        self.printTorques = False
        self.printRewards = False
        self.printActions = False
        self.balance_PID = False
        self.swingFoot = 'Right'

        self.framenum = 0
        self.framenum2 = 1
        self.framenum_left = self.framenum_right = 0
        self.prev_a = np.zeros(23,)
        self.init_count = 0
        self.balance = False
        self.trainRelay = False
        self.firstPass =  False
        self.qpos_node0 = np.zeros(29,)
        self.qpos_node1 = np.zeros(29,)
        self.qpos_node2 = np.zeros(29,)
        self.qpos_node3 = np.zeros(29,)

        #prefix = '../../Balance_getup/'
        # prefix = None
        # with open("rarm_endeffector_jump.txt","rb") as fp:
        #     self.rarm_endeffector = np.loadtxt(fp)

        # with open("larm_endeffector_jump.txt","rb") as fp:
        #     self.larm_endeffector = np.loadtxt(fp)

        # with open("lfoot_endeffector_jump.txt","rb") as fp:
        #     self.lfoot_endeffector = np.loadtxt(fp)

        # with open("rfoot_endeffector_jump.txt",'rb') as fp:
        #     self.rfoot_endeffector = np.loadtxt(fp)

        # with open("com_jump.txt",'rb') as fp:
        #     self.com = np.loadtxt(fp)
        # with open("jump.txt","rb") as fp:
        #     self.MotionPositions = np.loadtxt(fp)

        # with open("jumpvel.txt","rb") as fp:
        #     self.MotionVelocities = np.loadtxt(fp)


        dir_prefix = os.path.dirname(os.path.realpath(__file__)) + "/"
        skel_prefix = dir_prefix + "assets/skel/"
        mocap_prefix = dir_prefix + "assets/mocap/jump/"

        with open(mocap_prefix + "rarm_endeffector.txt","rb") as fp:
            self.rarm_endeffector = np.loadtxt(fp)

        with open(mocap_prefix + "larm_endeffector.txt","rb") as fp:
            self.larm_endeffector = np.loadtxt(fp)

        with open(mocap_prefix + "lfoot_endeffector.txt","rb") as fp:
            self.lfoot_endeffector = np.loadtxt(fp)

        with open(mocap_prefix + "rfoot_endeffector.txt",'rb') as fp:
            self.rfoot_endeffector = np.loadtxt(fp)

        with open(mocap_prefix + "com.txt",'rb') as fp:
            self.com = np.loadtxt(fp)
        with open(mocap_prefix + "positions.txt","rb") as fp:
            self.MotionPositions = np.loadtxt(fp)

        with open(mocap_prefix + "velocities.txt","rb") as fp:
            self.MotionVelocities = np.loadtxt(fp)

        self.num_frames = len(self.MotionPositions)

        self.prevdq = np.zeros(29,)
        self.tau = np.zeros(29,)
        self.ndofs = 29
        self.target = np.zeros(self.ndofs,)
        self.init = np.zeros(self.ndofs,)
        self.edot = np.zeros(self.ndofs,)
        self.preverror = np.zeros(self.ndofs,)
        self.angle_buf = np.zeros(2)
        self.stance = None
        self.target_vel = 1.0
        for i in range(6, self.ndofs):
            self.preverror[i] = (self.init[i] - self.target[i])
        self.t = 0

        dart_env.DartEnv.__init__(self, [skel_prefix + 'kima_original.skel'],
                                  32, obs_dim, self.control_bounds, disableViewer=False)

        self.robot_skeleton.set_self_collision_check(True)

        utils.EzPickle.__init__(self)

    def transformActions(self,actions):

        joint_targets = np.zeros(23,)

        # Left thigh
        lthigh = actions[:4]
        euler_lthigh = angle_axis2euler(theta=lthigh[0],vector=lthigh[1:])
        joint_targets[0] = euler_lthigh[2]
        joint_targets[1] = euler_lthigh[1]
        joint_targets[2] = euler_lthigh[0]

        ###### Left Knee
        joint_targets[3] = actions[4]

        ### left foot
        lfoot = actions[5:9]
        euler_lfoot = angle_axis2euler(theta=lfoot[0],vector=lfoot[1:])
        joint_targets[4] = euler_lfoot[2]
        joint_targets[5] = euler_lfoot[0]

        # right thigh
        rthigh = actions[9:13]
        euler_rthigh = angle_axis2euler(theta=rthigh[0],vector=rthigh[1:])
        joint_targets[6] = euler_rthigh[2]
        joint_targets[7] = euler_rthigh[1]
        joint_targets[8] = euler_rthigh[0]

        ###### right Knee
        joint_targets[9] = actions[13]

        ### right foot
        rfoot = actions[14:18]
        euler_rfoot = angle_axis2euler(theta=rfoot[0],vector=rfoot[1:])
        joint_targets[10] = euler_rfoot[2]
        joint_targets[11] = euler_rfoot[0]

        ###thorax
        thorax = actions[18:22]
        euler_thorax = angle_axis2euler(theta=thorax[0],vector=thorax[1:])
        joint_targets[12] = euler_thorax[2]
        joint_targets[13] = euler_thorax[1]
        joint_targets[14] = euler_thorax[0]

        #### l upper arm
        l_arm = actions[22:26]
        euler_larm = angle_axis2euler(theta=l_arm[0],vector=l_arm[1:])
        joint_targets[15] = euler_larm[2]
        joint_targets[16] = euler_larm[1]
        joint_targets[17] = euler_larm[0]

        ## l elbow
        joint_targets[18] = actions[26]

        ## r upper arm
        r_arm = actions[27:31]
        euler_rarm = angle_axis2euler(theta=r_arm[0],vector=r_arm[1:])
        joint_targets[19] = euler_rarm[2]
        joint_targets[20] = euler_rarm[1]
        joint_targets[21] = euler_rarm[0]

        ###r elbow
        joint_targets[22] = actions[31]

        return joint_targets

    def quat_reward(self, skel, framenum):

        quaternion_difference = []

        #### lthigh
        lthigh_euler = skel.q[6:9]
        lthigh_mocap = self.MotionPositions[framenum,6:9]
        quat_lthigh = euler2quat(z=lthigh_euler[2],
                                 y=lthigh_euler[1],
                                 x=lthigh_euler[0])
        quat_lthigh_mocap = euler2quat(z=lthigh_mocap[2],
                                       y=lthigh_mocap[1],
                                       x=lthigh_mocap[0])
        lthigh_diff = mult(inverse(quat_lthigh_mocap),quat_lthigh)
        scalar_lthigh = 2*np.arccos(lthigh_diff[0])
        quaternion_difference.append(scalar_lthigh)

        ##### lknee
        lknee_euler = skel.q[9]
        lknee_mocap = self.MotionPositions[framenum,9]
        quat_lknee = euler2quat(z=0.,y=0.,x=lknee_euler)
        quat_lknee_mocap = euler2quat(z=0.,y=0.,x=lknee_mocap)
        lknee_diff = mult(inverse(quat_lknee_mocap),quat_lknee)
        scalar_lknee = 2*np.arccos(lknee_diff[0])
        quaternion_difference.append(scalar_lknee)

        #### lfoot
        lfoot_euler = skel.q[10:12]
        lfoot_mocap = self.MotionPositions[framenum,10:12]
        quat_lfoot = euler2quat(z=lfoot_euler[1],y=0.,x=lfoot_euler[0])
        quat_lfoot_mocap = euler2quat(z=lfoot_mocap[1],y=0.,x=lfoot_mocap[0])
        lfoot_diff = mult(inverse(quat_lfoot_mocap),quat_lfoot)
        scalar_lfoot = 2*np.arccos(lfoot_diff[0])
        quaternion_difference.append(scalar_lfoot)

        #### rthigh
        rthigh_euler = skel.q[12:15]
        rthigh_mocap = self.MotionPositions[framenum,12:15]
        quat_rthigh = euler2quat(z=rthigh_euler[2],
                                 y=rthigh_euler[1],
                                 x=rthigh_euler[0])
        quat_rthigh_mocap = euler2quat(z=rthigh_mocap[2],
                                       y=rthigh_mocap[1],
                                       x=rthigh_mocap[0])
        rthigh_diff = mult(inverse(quat_rthigh_mocap),quat_rthigh)
        scalar_rthigh = 2*np.arccos(rthigh_diff[0])
        quaternion_difference.append(scalar_rthigh)

        ##### rknee
        rknee_euler = skel.q[15]
        rknee_mocap = self.MotionPositions[framenum,15]
        quat_rknee = euler2quat(z=0.,y=0.,x=rknee_euler)
        quat_rknee_mocap = euler2quat(z=0.,y=0.,x=rknee_mocap)
        rknee_diff = mult(inverse(quat_rknee_mocap),quat_rknee)
        scalar_rknee = 2*np.arccos(rknee_diff[0])
        quaternion_difference.append(scalar_rknee)

        #### rfoot
        rfoot_euler = skel.q[16:18]
        rfoot_mocap = self.MotionPositions[framenum,16:18]
        quat_rfoot = euler2quat(z=rfoot_euler[1],y=0.,x=rfoot_euler[0])
        quat_rfoot_mocap = euler2quat(z=rfoot_mocap[1],y=0.,x=rfoot_mocap[0])
        rfoot_diff = mult(inverse(quat_rfoot_mocap),quat_rfoot)
        scalar_rfoot = 2*np.arccos(rfoot_diff[0])
        quaternion_difference.append(scalar_rfoot)

        ### Thorax
        scalar_thoraxx = skel.q[18] - self.MotionPositions[framenum,18]
        scalar_thoraxy = skel.q[19] - self.MotionPositions[framenum,19]
        scalar_thoraxz = skel.q[20] - self.MotionPositions[framenum,20]
        quaternion_difference.append(scalar_thoraxx)
        quaternion_difference.append(scalar_thoraxy)
        quaternion_difference.append(scalar_thoraxz)

        #### l upper arm
        larm_euler = skel.q[21:24]
        larm_mocap = self.MotionPositions[framenum,21:24]
        quat_larm = euler2quat(z=larm_euler[2],y=larm_euler[1],x=larm_euler[0])
        quat_larm_mocap = euler2quat(z=larm_mocap[2],
                                     y=larm_mocap[1],
                                     x=larm_mocap[0])
        larm_diff = mult(inverse(quat_larm_mocap),quat_larm)
        scalar_larm = 2*np.arccos(larm_diff[0])
        quaternion_difference.append(scalar_larm)

        ##### l elbow
        lelbow_euler = skel.q[24]
        lelbow_mocap = self.MotionPositions[framenum,24]
        quat_lelbow = euler2quat(z=0.,y=0.,x=lelbow_euler)
        quat_lelbow_mocap = euler2quat(z=0.,y=0.,x=lelbow_mocap)
        lelbow_diff = mult(inverse(quat_lelbow_mocap),quat_lelbow)
        scalar_lelbow = 2*np.arccos(lelbow_diff[0])
        quaternion_difference.append(scalar_lelbow)

        #### r upper arm
        rarm_euler = skel.q[25:28]
        rarm_mocap = self.MotionPositions[framenum,25:28]
        quat_rarm = euler2quat(z=rarm_euler[2],y=rarm_euler[1],x=rarm_euler[0])
        quat_rarm_mocap = euler2quat(z=rarm_mocap[2],
                                     y=rarm_mocap[1],
                                     x=rarm_mocap[0])
        rarm_diff = mult(inverse(quat_rarm_mocap),quat_rarm)
        scalar_rarm = 2*np.arccos(rarm_diff[0])
        quaternion_difference.append(scalar_rarm)

        ##### r elbow
        relbow_euler = skel.q[28]
        relbow_mocap = self.MotionPositions[framenum,28]
        quat_relbow = euler2quat(z=0.,y=0.,x=relbow_euler)
        quat_relbow_mocap = euler2quat(z=0.,y=0.,x=relbow_mocap)
        relbow_diff = mult(inverse(quat_relbow_mocap),quat_relbow)
        scalar_relbow = 2*np.arccos(relbow_diff[0])
        quaternion_difference.append(scalar_relbow)

        return np.exp(-2*np.sum(np.square(quaternion_difference)))

    def advance(self, a):

        clamped_control = np.array(a)

        self.tau = np.zeros(self.robot_skeleton.ndofs)

        self.target[6:] = self.transformActions(clamped_control) \
                          + self.MotionPositions[self.framenum,6:]

        for i in range(4):
            self.tau[6:] = self.PID(self.robot_skeleton, self.target)

            self.robot_skeleton.set_forces(self.tau)
            self.dart_world.step()

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

        for i in range(6,self.ndofs):
            if torques[i] > torqueLimits[i-6]:
                torques[i] = torqueLimits[i-6]
            if torques[i] < -torqueLimits[i-6]:
                torques[i] = -torqueLimits[i-6]

        return torques

    def PID(self, skel, target):
        #print("#########################################################################3")
        self.kp = np.array([250]*23)#350
        self.kd = np.array([0.005]*23)

        self.kp[0] = 600+25 #1000#
        self.kp[3] = 225+25 #0#450#
        self.kp[9] = 225+25
        self.kp[10] = 200
        self.kp[16] = 200
        self.kp[[1,2]] = 150
        self.kp[[7,8]] = 150
        self.kp[6] =600+25
        #self.kd[[4,510,11]] = 0.005
        self.kp[15:] = 155
        self.kd[15:]= 0.05

        self.kp = [item/2  for item in self.kp]
        self.kd = [item/2  for item in self.kd]

        q = skel.q
        qdot = skel.dq
        tau = np.zeros((self.ndofs,))
        for i in range(6, self.ndofs):
            #print(q.shape)
            self.edot[i] = ((q[i] - target[i]) -
                self.preverror[i]) / self.dt
            tau[i] = -self.kp[i - 6] * \
                (q[i] - target[i]) - \
                self.kd[i - 6] *qdot[i]
            self.preverror[i] = (q[i] - target[i])

        torqs = self.ClampTorques(tau)

        return torqs[6:]

    def com_reward(self, skel, framenum):
        return np.exp(-40*np.sum(np.square(self.com[framenum,:] \
                                           - skel.bodynodes[0].com())))

    def ee_reward(self, skel, framenum):

        point_rarm = [0.,-0.60,-0.15]
        point_larm = [0.,-0.60,-0.15]
        point_rfoot = [0.,0.,-0.20]
        point_lfoot = [0.,0.,-0.20]

        global_rarm = skel.bodynodes[16].to_world(point_rarm)
        global_larm = skel.bodynodes[13].to_world(point_larm)
        global_lfoot = skel.bodynodes[4].to_world(point_lfoot)
        global_rfoot = skel.bodynodes[7].to_world(point_rfoot)

        rarm_term = np.sum(np.square(self.rarm_endeffector[framenum,:] \
                                     - global_rarm))
        larm_term = np.sum(np.square(self.larm_endeffector[framenum,:] \
                                     - global_larm))
        rfoot_term = np.sum(np.square(self.rfoot_endeffector[framenum,:] \
                                      - global_rfoot))
        lfoot_term = np.sum(np.square(self.lfoot_endeffector[framenum,:] \
                                      - global_lfoot))

        return np.exp(-40*(rarm_term + larm_term + \
                           rfoot_term + lfoot_term))

    def vel_reward(self, skel, framenum):

        Joint_weights = np.ones(23,)
        Joint_weights[[0,3,6,9,16,20,10,16]] = 10
        Weight_matrix = np.diag(Joint_weights)

        vel_diff = self.MotionVelocities[framenum,6:] - skel.dq[6:]

        vel_pen = np.sum(vel_diff.T*Weight_matrix*vel_diff)

        return 1*np.asarray(np.exp(-1e-1*vel_pen))

    def reward(self, skel, framenum):

        R_ee = self.ee_reward(skel, framenum)
        R_com = self.com_reward(skel, framenum)
        R_vel = self.vel_reward(skel, framenum)
        R_quat = self.quat_reward(skel, framenum)

        return 0.10*R_ee + 0.10*R_vel + 0.25*R_com + 1.65*R_quat

    def should_terminate(self, skel, obs):

        height = skel.bodynodes[0].com()[1]

        return not (np.isfinite(obs).all()
                    and (np.abs(obs[2:]) < 200).all()
                    and (height > -0.70) and (height < 0.40)
                    and (abs(skel.q[4]) < 0.30)
                    and (abs(skel.q[5]) < 0.50)
                    and (skel.q[3] > -0.4)
                    and (skel.q[3] < 0.3))

    def _step(self, a):


        ##################################################################
        # Warning! Duplicated code

        point_rarm = [0.,-0.60,-0.15]
        point_larm = [0.,-0.60,-0.15]
        point_rfoot = [0.,0.,-0.20]
        point_lfoot = [0.,0.,-0.20]

        global_rarm=self.robot_skeleton.bodynodes[16].to_world(point_rarm)
        global_larm=self.robot_skeleton.bodynodes[13].to_world(point_larm)
        global_lfoot=self.robot_skeleton.bodynodes[4].to_world(point_lfoot)
        global_rfoot=self.robot_skeleton.bodynodes[7].to_world(point_rfoot)

        # End duplicated code
        ##################################################################

        self.dart_world.set_text = []
        self.dart_world.y_scale = np.clip(a[6],-2,2)
        self.dart_world.plot = False
        posbefore = self.robot_skeleton.bodynodes[0].com()[0]

        self.advance(a)

        self.dart_world.contact_point = []
        self.dart_world.contact_color = 'red'
        self.dart_world.contact_point.append(global_rarm)
        self.dart_world.contact_point.append(global_larm)
        self.dart_world.contact_point.append(global_rfoot)
        self.dart_world.contact_point.append(global_lfoot)

        posafter = self.robot_skeleton.bodynodes[0].com()[0]

        vel = (posafter - posbefore) / self.dt

        R_total = self.reward(self.robot_skeleton, self.framenum)

        contacts = self.dart_world.collision_result.contacts
        head_flag = False
        for item in contacts:
            if item.skel_id1 == 0:
                if self.robot_skeleton.bodynodes[item.bodynode_id2].name == "head":
                    head_flag = True

        s = self.state_vector()
        done = self.should_terminate(self.robot_skeleton,
                                     self.state_vector())

        if done:
            R_total = 0.

        ob = self._get_obs()

        if head_flag:
            reward = 0.
            done = True

        ob = self._get_obs()
        self.framenum += 1
        if self.framenum >= self.num_frames-1:
            done = True

        return ob, R_total, done, {}

    def _get_obs(self):


        phi = np.array([self.framenum/self.MotionPositions.shape[0]])
        links = [2,3,4,5,6,7,12,13,15,16]
        # observation for left leg thigh##################################################
        RelPos_lthigh = self.robot_skeleton.bodynodes[2].com() - self.robot_skeleton.bodynodes[0].com()
        state = copy.deepcopy(RelPos_lthigh)
        quat_lthigh = euler2quat(z=self.robot_skeleton.q[8],y=self.robot_skeleton.q[7],x=self.robot_skeleton.q[6])
        state = np.concatenate((state,quat_lthigh))
        LinVel_lthigh = self.robot_skeleton.bodynodes[2].dC
        state = np.concatenate((state,LinVel_lthigh))
        state = np.concatenate((state,self.robot_skeleton.dq[6:9]))
        ################################################################3
        RelPos_lknee = self.robot_skeleton.bodynodes[3].com() - self.robot_skeleton.bodynodes[0].com()
        state = np.concatenate((state,RelPos_lknee))
        quat_lknee = euler2quat(z=0.,y=0.,x=self.robot_skeleton.q[9])
        state = np.concatenate((state,quat_lknee))
        LinVel_lknee = self.robot_skeleton.bodynodes[3].dC
        state = np.concatenate((state,LinVel_lknee))
        state = np.concatenate((state,np.array([self.robot_skeleton.dq[9]])))
        #######################################################################3
        RelPos_lfoot = self.robot_skeleton.bodynodes[4].com() - self.robot_skeleton.bodynodes[0].com()
        state = np.concatenate((state,RelPos_lfoot))
        quat_lfoot = euler2quat(z=self.robot_skeleton.q[11],y=0.,x=self.robot_skeleton.q[10])
        state = np.concatenate((state,quat_lfoot))
        LinVel_lfoot = self.robot_skeleton.bodynodes[4].dC
        state = np.concatenate((state,LinVel_lfoot))
        state = np.concatenate((state,self.robot_skeleton.dq[10:12]))
        #######################################################################3
        RelPos_rthigh = self.robot_skeleton.bodynodes[5].com() - self.robot_skeleton.bodynodes[0].com()
        state = np.concatenate((state,RelPos_rthigh))
        quat_rthigh = euler2quat(z=self.robot_skeleton.q[14],y=self.robot_skeleton.q[13],x=self.robot_skeleton.q[12])
        state = np.concatenate((state,quat_rthigh))
        LinVel_rthigh = self.robot_skeleton.bodynodes[5].dC
        state = np.concatenate((state,LinVel_rthigh))
        state = np.concatenate((state,self.robot_skeleton.dq[12:15]))
        ###############################################################################3
        RelPos_rknee = self.robot_skeleton.bodynodes[6].com() - self.robot_skeleton.bodynodes[0].com()
        state = np.concatenate((state,RelPos_rknee))
        quat_rknee = euler2quat(z=0.,y=0.,x=self.robot_skeleton.q[15])
        state = np.concatenate((state,quat_rknee))
        LinVel_rknee = self.robot_skeleton.bodynodes[6].dC
        state = np.concatenate((state,LinVel_rknee))
        state = np.concatenate((state,np.array([self.robot_skeleton.dq[15]])))
        ################################################################################3
        RelPos_rfoot = self.robot_skeleton.bodynodes[7].com() - self.robot_skeleton.bodynodes[0].com()
        state = np.concatenate((state,RelPos_rfoot))
        quat_rfoot = euler2quat(z=self.robot_skeleton.q[17],y=0.,x=self.robot_skeleton.q[16])
        state = np.concatenate((state,quat_rfoot))
        LinVel_rfoot = self.robot_skeleton.bodynodes[7].dC
        state = np.concatenate((state,LinVel_rfoot))
        state = np.concatenate((state,self.robot_skeleton.dq[16:18]))
        ###########################################################
        RelPos_larm = self.robot_skeleton.bodynodes[12].com() - self.robot_skeleton.bodynodes[0].com()
        state = np.concatenate((state,RelPos_larm))
        quat_larm = euler2quat(z=self.robot_skeleton.q[23],y=self.robot_skeleton.q[22],x=self.robot_skeleton.q[21])
        state = np.concatenate((state,quat_larm))
        LinVel_larm = self.robot_skeleton.bodynodes[12].dC
        state = np.concatenate((state,LinVel_larm))
        state = np.concatenate((state,self.robot_skeleton.dq[21:24]))
        ##############################################################
        RelPos_lelbow = self.robot_skeleton.bodynodes[13].com() - self.robot_skeleton.bodynodes[0].com()
        state = np.concatenate((state,RelPos_lelbow))
        quat_lelbow = euler2quat(z=0.,y=0.,x=self.robot_skeleton.q[24])
        state = np.concatenate((state,quat_lelbow))
        LinVel_lelbow = self.robot_skeleton.bodynodes[13].dC
        state = np.concatenate((state,LinVel_lelbow))
        state = np.concatenate((state,np.array([self.robot_skeleton.dq[24]])))
        ################################################################
        RelPos_rarm = self.robot_skeleton.bodynodes[15].com() - self.robot_skeleton.bodynodes[0].com()
        state = np.concatenate((state,RelPos_rarm))
        quat_rarm = euler2quat(z=self.robot_skeleton.q[27],y=self.robot_skeleton.q[26],x=self.robot_skeleton.q[25])
        state = np.concatenate((state,quat_rarm))
        LinVel_rarm = self.robot_skeleton.bodynodes[15].dC
        state = np.concatenate((state,LinVel_rarm))
        state = np.concatenate((state,self.robot_skeleton.dq[25:28]))
        #################################################################3
        RelPos_relbow = self.robot_skeleton.bodynodes[16].com() - self.robot_skeleton.bodynodes[0].com()
        state = np.concatenate((state,RelPos_relbow))
        quat_relbow = euler2quat(z=0.,y=0.,x=self.robot_skeleton.q[28])
        state = np.concatenate((state,quat_relbow))
        LinVel_relbow = self.robot_skeleton.bodynodes[16].dC
        state = np.concatenate((state,LinVel_relbow))
        state = np.concatenate((state,np.array([self.robot_skeleton.dq[28]])))
        state = np.concatenate((state,self.robot_skeleton.q[18:21],self.robot_skeleton.dq[18:21],phi))
        ##################################################################



        return state

    def get_random_framenum(self):
        return np.random.randint(low=1,
                                 high=self.num_frames - 1,
                                 size=1)[0]

    def reset_model(self):

        self.dart_world.reset()

        rand_start = self.get_random_framenum()
        self.framenum = rand_start

        qpos = self.MotionPositions[rand_start,:].reshape(29,) \
               + self.np_random.uniform(low=-0.0050,
                                        high=.0050,
                                        size=self.robot_skeleton.ndofs)

        qvel = self.MotionVelocities[rand_start,:].reshape(29,) \
               + self.np_random.uniform(low=-0.0050,
                                        high=.0050,
                                        size=self.robot_skeleton.ndofs)

        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = 5.0
            self._get_viewer().scene.tb.trans[2] = -7.5
            self._get_viewer().scene.tb.trans[1] = 0.0
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


def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)
