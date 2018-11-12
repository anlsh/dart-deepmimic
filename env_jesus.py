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

        self.count = 0
        self.count2 = 1
        self.count_left = self.count_right = 0
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
        #     self.WalkPositions = np.loadtxt(fp)

        # with open("jumpvel.txt","rb") as fp:
        #     self.WalkVelocities = np.loadtxt(fp)


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
            self.WalkPositions = np.loadtxt(fp)

        with open(mocap_prefix + "velocities.txt","rb") as fp:
            self.WalkVelocities = np.loadtxt(fp)

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
        #print("LARM",actions.shape)
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

    def ComputeReward(self,):
        quaternion_difference = []
        #### lthigh
        lthigh_euler = self.robot_skeleton.q[6:9]
        lthigh_mocap = self.WalkPositions[self.count,6:9]
        quat_lthigh = euler2quat(z=lthigh_euler[2],y=lthigh_euler[1],x=lthigh_euler[0])
        quat_lthigh_mocap = euler2quat(z=lthigh_mocap[2],y=lthigh_mocap[1],x=lthigh_mocap[0])
        lthigh_diff = mult(inverse(quat_lthigh_mocap),quat_lthigh)
        scalar_lthigh = 2*np.arccos(lthigh_diff[0])
        quaternion_difference.append(scalar_lthigh)
        #print("scaler",scalar_lthigh)
        ##### lknee
        lknee_euler = self.robot_skeleton.q[9]
        lknee_mocap = self.WalkPositions[self.count,9]
        quat_lknee = euler2quat(z=0.,y=0.,x=lknee_euler)
        quat_lknee_mocap = euler2quat(z=0.,y=0.,x=lknee_mocap)
        lknee_diff = mult(inverse(quat_lknee_mocap),quat_lknee)
        scalar_lknee = 2*np.arccos(lknee_diff[0])
        quaternion_difference.append(scalar_lknee)
        #### lfoot
        lfoot_euler = self.robot_skeleton.q[10:12]
        lfoot_mocap = self.WalkPositions[self.count,10:12]
        quat_lfoot = euler2quat(z=lfoot_euler[1],y=0.,x=lfoot_euler[0])
        quat_lfoot_mocap = euler2quat(z=lfoot_mocap[1],y=0.,x=lfoot_mocap[0])
        lfoot_diff = mult(inverse(quat_lfoot_mocap),quat_lfoot)
        scalar_lfoot = 2*np.arccos(lfoot_diff[0])
        quaternion_difference.append(scalar_lfoot)
        #### rthigh
        rthigh_euler = self.robot_skeleton.q[12:15]
        rthigh_mocap = self.WalkPositions[self.count,12:15]
        quat_rthigh = euler2quat(z=rthigh_euler[2],y=rthigh_euler[1],x=rthigh_euler[0])
        quat_rthigh_mocap = euler2quat(z=rthigh_mocap[2],y=rthigh_mocap[1],x=rthigh_mocap[0])
        rthigh_diff = mult(inverse(quat_rthigh_mocap),quat_rthigh)
        scalar_rthigh = 2*np.arccos(rthigh_diff[0])
        quaternion_difference.append(scalar_rthigh)
        #print("scaler",scalar_lthigh)
        ##### rknee
        rknee_euler = self.robot_skeleton.q[15]
        rknee_mocap = self.WalkPositions[self.count,15]
        quat_rknee = euler2quat(z=0.,y=0.,x=rknee_euler)
        quat_rknee_mocap = euler2quat(z=0.,y=0.,x=rknee_mocap)
        rknee_diff = mult(inverse(quat_rknee_mocap),quat_rknee)
        scalar_rknee = 2*np.arccos(rknee_diff[0])
        quaternion_difference.append(scalar_rknee)
        #### rfoot
        rfoot_euler = self.robot_skeleton.q[16:18]
        rfoot_mocap = self.WalkPositions[self.count,16:18]
        quat_rfoot = euler2quat(z=rfoot_euler[1],y=0.,x=rfoot_euler[0])
        quat_rfoot_mocap = euler2quat(z=rfoot_mocap[1],y=0.,x=rfoot_mocap[0])
        rfoot_diff = mult(inverse(quat_rfoot_mocap),quat_rfoot)
        scalar_rfoot = 2*np.arccos(rfoot_diff[0])
        quaternion_difference.append(scalar_rfoot)

        scalar_thoraxx = self.robot_skeleton.q[18] - self.WalkPositions[self.count,18]
        quaternion_difference.append(scalar_thoraxx)
        scalar_thoraxy = self.robot_skeleton.q[19] - self.WalkPositions[self.count,19]
        quaternion_difference.append(scalar_thoraxy)
        scalar_thoraxz = self.robot_skeleton.q[20] - self.WalkPositions[self.count,20]
        quaternion_difference.append(scalar_thoraxz)
        #### l upper arm
        larm_euler = self.robot_skeleton.q[21:24]
        larm_mocap = self.WalkPositions[self.count,21:24]
        quat_larm = euler2quat(z=larm_euler[2],y=larm_euler[1],x=larm_euler[0])
        quat_larm_mocap = euler2quat(z=larm_mocap[2],y=larm_mocap[1],x=larm_mocap[0])
        larm_diff = mult(inverse(quat_larm_mocap),quat_larm)
        scalar_larm = 2*np.arccos(larm_diff[0])
        quaternion_difference.append(scalar_larm)
        #print("scaler",scalar_lthigh)
        ##### l elbow
        lelbow_euler = self.robot_skeleton.q[24]
        lelbow_mocap = self.WalkPositions[self.count,24]
        quat_lelbow = euler2quat(z=0.,y=0.,x=lelbow_euler)
        quat_lelbow_mocap = euler2quat(z=0.,y=0.,x=lelbow_mocap)
        lelbow_diff = mult(inverse(quat_lelbow_mocap),quat_lelbow)
        scalar_lelbow = 2*np.arccos(lelbow_diff[0])
        quaternion_difference.append(scalar_lelbow)
        #### r upper arm
        rarm_euler = self.robot_skeleton.q[25:28]
        rarm_mocap = self.WalkPositions[self.count,25:28]
        quat_rarm = euler2quat(z=rarm_euler[2],y=rarm_euler[1],x=rarm_euler[0])
        quat_rarm_mocap = euler2quat(z=rarm_mocap[2],y=rarm_mocap[1],x=rarm_mocap[0])
        rarm_diff = mult(inverse(quat_rarm_mocap),quat_rarm)
        scalar_rarm = 2*np.arccos(rarm_diff[0])
        quaternion_difference.append(scalar_rarm)
        #print("scaler",scalar_lthigh)
        ##### r elbow
        relbow_euler = self.robot_skeleton.q[28]
        relbow_mocap = self.WalkPositions[self.count,28]
        quat_relbow = euler2quat(z=0.,y=0.,x=relbow_euler)
        quat_relbow_mocap = euler2quat(z=0.,y=0.,x=relbow_mocap)
        relbow_diff = mult(inverse(quat_relbow_mocap),quat_relbow)
        scalar_relbow = 2*np.arccos(relbow_diff[0])
        quaternion_difference.append(scalar_relbow)

        quat_reward = np.exp(-2*np.sum(np.square(quaternion_difference)))
        #print("reward",quat_reward)

        return quat_reward


    def advance(self, a):




        clamped_control = np.array(a)



        self.tau = np.zeros(self.robot_skeleton.ndofs)
        trans = np.zeros(6,)


        #if self.swingFoot == "Right":
        self.target[6:]=  self.transformActions(clamped_control) + self.WalkPositions[self.count,6:] #*self.action_scale# + self.ref_trajectory_right[self.count_right,6:]# +



        for i in range(4):
            self.tau[6:] = self.PID()



            if self.dumpTorques:
                with open("torques.txt","ab") as fp:
                    np.savetxt(fp,np.array([self.tau]),fmt='%1.5f')

            if self.dumpActions:
                with open("targets_from_net.txt",'ab') as fp:
                    np.savetxt(fp,np.array([[self.target[6],self.robot_skeleton.q[6]]]),fmt='%1.5f')


            self.robot_skeleton.set_forces(self.tau)
            #print("torques",self.tau[22])
            self.dart_world.step()



    def ClampTorques(self,torques):
        torqueLimits = np.array([150.0*5,80.*3,80.*3,100.*5,80.*5,60.,150.0*5,80.*3,80.*3,100.*5,80.*5,60.,150.*5,150.*5,150.*5,10.,5.,5.,5.,10.,5.,5,5.])*2
        #print("tau",torqueLimits[4])
        for i in range(6,self.ndofs):
            if torques[i] > torqueLimits[i-6]:
                torques[i] = torqueLimits[i-6]
            if torques[i] < -torqueLimits[i-6]:
                torques[i] = -torqueLimits[i-6]

        return torques

    def PID(self):
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

        q = self.robot_skeleton.q
        qdot = self.robot_skeleton.dq
        tau = np.zeros((self.ndofs,))
        for i in range(6, self.ndofs):
            #print(q.shape)
            self.edot[i] = ((q[i] - self.target[i]) -
                self.preverror[i]) / self.dt
            tau[i] = -self.kp[i - 6] * \
                (q[i] - self.target[i]) - \
                self.kd[i - 6] *qdot[i]
            self.preverror[i] = (q[i] - self.target[i])

        torqs = self.ClampTorques(tau)

        return torqs[6:]


    def _step(self, a):

        self.dart_world.set_text = []
        self.dart_world.y_scale = np.clip(a[6],-2,2)
        self.dart_world.plot = False
        count_str = "count :"+str(self.count)
        a_from_net = "a[6] : %f and a[12] : %f"%(a[16],a[20])
        #self.dart_world.set_text.append(a_from_net)
        self.dart_world.set_text.append(count_str)
        posbefore = self.robot_skeleton.bodynodes[0].com()[0]


        if self.duplicate:
            dupq = self.robot_skeleton.q
            dupq[0] = 1.0
            self.dupSkel.set_positions(dupq)
            self.dupskel.set_velocities(np.zeros(self.robot_skeleton.q.shape[0],))


        self.advance(a)
        if self.dumpActions:
            with open("a_from_net.txt","ab") as fp:
                np.savetxt(fp,np.array([a]),fmt='%1.5f')


        #print("torques",self.tau[[6,12]])
        point_rarm = [0.,-0.60,-0.15]
        point_larm = [0.,-0.60,-0.15]
        point_rfoot = [0.,0.,-0.20]
        point_lfoot = [0.,0.,-0.20]

        global_rarm = self.robot_skeleton.bodynodes[16].to_world(point_rarm)

        global_larm = self.robot_skeleton.bodynodes[13].to_world(point_larm)
        global_lfoot = self.robot_skeleton.bodynodes[4].to_world(point_lfoot)
        global_rfoot = self.robot_skeleton.bodynodes[7].to_world(point_rfoot)



        self.dart_world.contact_point = []
        self.dart_world.contact_color = 'red'
        self.dart_world.contact_point.append(global_rarm)
        self.dart_world.contact_point.append(global_larm)
        self.dart_world.contact_point.append(global_rfoot)
        self.dart_world.contact_point.append(global_lfoot)



        #print(self.swingFoot)
        posafter = self.robot_skeleton.bodynodes[0].com()[0]
        height = self.robot_skeleton.bodynodes[0].com()[1]
        side_deviation = self.robot_skeleton.bodynodes[0].com()[2]

        upward_world = self.robot_skeleton.bodynodes[0].to_world(np.array([1, 0, 0]))
        init_axis = [  1.22906906e+00,  -3.59145383e-02 ,  1.24547289e-04]
        ang_cos_uwd = np.dot(upward_world,init_axis)/(np.linalg.norm(upward_world)*np.linalg.norm(init_axis))
        ang_cos_uwd = np.arccos(ang_cos_uwd)
        if np.isnan(ang_cos_uwd):
            ang_cos_uwd = 0.
        self.angle_buf[0] = self.angle_buf[1]
        self.angle_buf[1] = ang_cos_uwd

        or_vel = (self.angle_buf[1] - self.angle_buf[0])/0.016

        contacts = self.dart_world.collision_result.contacts

        if contacts == []:
            self.switch = 0

        if contacts != []:
            self.switch = 1


        self.ComputeReward()

        total_force_mag = 0
        for contact in contacts:


            force = np.sum(contact.force[[1,2]])



            total_force_mag += force
            data = np.zeros(11,)
            data[:10] = contact.state
            data[10] = self.count


        if self.dumpCOM:
            with open("COM_fromPolicy_walk.txt","ab") as fp:
                np.savetxt(fp,np.asarray(self.robot_skeleton.q[:3]),fmt="%1.5f")

            with open("Joint_fromPolicy_walk.txt","ab") as fp:
                np.savetxt(fp,np.asarray(self.robot_skeleton.q),fmt="%1.5f")



        alive_bonus = 4
        vel = (posafter - posbefore) / self.dt
        #print("a shape",a[-1])
        action_pen = np.sqrt(np.square(a[:23]).sum())

        reward = 0


        W_joint = 2.
        W_joint_vel = 2.
        W_trans = 2.
        W_orient = 1.

        #W_theta = 0.5
        W_joint_vel = 1.
        W_trans_vel = 0.5
        W_orient_vel = 0.5
        W_balance = 5.0

        Joint_weights = np.ones(23,)#
        Joint_weights[[0,3,6,9,16,20,10,16]] = 10

        Weight_matrix = np.diag(Joint_weights)
        Weight_matrix_1 = np.diag(np.ones(4,))

        veldq = np.copy(self.robot_skeleton.dq)
        #print("ve",veldq)
        acc = (veldq - self.prevdq)/0.008


        com_height = np.square(0 - self.robot_skeleton.bodynodes[0].com()[1])
        com_height_reward = 10*np.exp(-5*com_height)
        right_foot = np.square(0 - self.robot_skeleton.bodynodes[7].com()[1])
        right_foot_reward = 10*np.exp(-5*right_foot)



        done = False


        rarm_term = np.sum(np.square(self.rarm_endeffector[self.count,:] - global_rarm))
        larm_term = np.sum(np.square(self.larm_endeffector[self.count,:] - global_larm))
        rfoot_term = np.sum(np.square(self.rfoot_endeffector[self.count,:] - global_rfoot))
        lfoot_term = np.sum(np.square(self.lfoot_endeffector[self.count,:] - global_lfoot))

        end_effector_reward = np.exp(-40*(rarm_term+larm_term+rfoot_term+lfoot_term))
        com_reward = np.exp(-40*np.sum(np.square(self.com[self.count,:] - self.robot_skeleton.bodynodes[0].com())))

        s = self.state_vector()







        joint_diff = self.WalkPositions[self.count,6:] - self.robot_skeleton.q[6:]#hmm[[6,9,12,15,22,26,10,16]] - self.robot_skeleton.q[[6,9,12,15,22,26,10,16]]
        #joint_diff_unimp = hmm[[7,8,13,14]] - self.robot_skeleton.q[[7,8,13,14]]
        joint_pen = np.sum(joint_diff.T*Weight_matrix*joint_diff)
        #joint_pen_unimp = np.sum(joint_diff_unimp.T*Weight_matrix_1*joint_diff_unimp)

        vel_diff = self.WalkVelocities[self.count,6:] - self.robot_skeleton.dq[6:]

        vel_pen = np.sum(vel_diff.T*Weight_matrix*vel_diff)

        node1_trans = np.array([0,-0.25,0])
        node1_root_orient = np.array([-np.pi/5,0,0])
        node0_trans = self.qpos_node0[:3]

        trans_pen = np.sum(np.square(node0_trans[:3] - self.robot_skeleton.q[:3]))
        trans_vel_pen = np.sum(np.square(np.zeros(3,) - self.robot_skeleton.dq[:3]))
        root_orient_pen = np.sum(np.square(self.WalkPositions[self.count,:3] - self.robot_skeleton.q[:3]))
        root_orient_vel =np.sum(np.square(self.WalkVelocities[self.count,:3] - self.robot_skeleton.dq[:3]))


        root_trans_term = 10/(1+ 100*trans_pen)#np.asarray(W_trans*np.exp(-10*trans_pen))
        #
        root_trans_vel = 100/(1+ 100*trans_vel_pen)#np.asarray(W_joint*np.exp(-10*trans_vel_pen))
        #
        joint_term = 1*np.asarray(np.exp(-2e-1*joint_pen))#np.asarray(W_joint*np.exp(-1e-2*joint_pen)) #100
        #joint_term_unimp = np.asarray(W_joint*np.exp(-joint_pen_unimp))
        #
        joint_vel_term = 1*np.asarray(np.exp(-1e-1*vel_pen))# W_joint_vel*np.exp(-1e-3*vel_pen)




        quat_term = self.ComputeReward()
        reward = 0.10*end_effector_reward + 0.10*joint_vel_term+ 0.25*com_reward+ 1.65*quat_term# + 0.75*body_orient + 0.45*body_orient_vel)/2.0 #+ 2.00*orient_term + 1.5*orient_vel                                                                                        # + contact_reward#  + joint_term + joint_vel_term #0.1*self.robot_skeleton.bodynodes[0].com()[1] + joint_term + joint_vel_term +
        eerew_str = "End Effector :"+str(end_effector_reward)
        self.dart_world.set_text.append(eerew_str)

        vel_str = "Joint Vel :"+str(joint_vel_term)
        self.dart_world.set_text.append(vel_str)

        com_str = "Com  :"+str(com_reward)
        self.dart_world.set_text.append(com_str)

        joint_str = "Joint :"+str(quat_term)
        self.dart_world.set_text.append(joint_str)


        lthigh_str = "Left Thigh target:"+str(self.WalkPositions[self.count,6])+" thigh Position :"+str(self.robot_skeleton.q[6])
        self.dart_world.set_text.append(lthigh_str)
        rthigh_str = "right Thigh target:"+str(self.WalkPositions[self.count,12])+" thigh Position :"+str(self.robot_skeleton.q[12])
        self.dart_world.set_text.append(rthigh_str)

        lthigh_torque = "Left Thigh torque:"+str(self.tau[6])
        self.dart_world.set_text.append(lthigh_torque)
        rthigh_torque = "right Thigh torque:"+str(self.tau[12])
        self.dart_world.set_text.append(rthigh_torque)

        com_vel = "com_vel:"+str(self.robot_skeleton.q[1])
        self.dart_world.set_text.append(com_vel)

        tar_vel = "tar_com_vel:"+str(self.WalkVelocities[self.count,1])
        self.dart_world.set_text.append(tar_vel)

        h = "height:"+str(height)
        self.dart_world.set_text.append(h)

        c = 0
        head_flag = False
        for item in contacts:
            #c = 0
            if item.skel_id1 == 0:
                if self.robot_skeleton.bodynodes[item.bodynode_id2].name == "head":
                    #print("Headddddddd")
                    head_flag = True
                if self.robot_skeleton.bodynodes[item.bodynode_id2].name == "l-lowerarm":
                    c+=1
                if self.robot_skeleton.bodynodes[item.bodynode_id2].name == "r-lowerarm":
                    #print("true")
                    c+=1
                if self.robot_skeleton.bodynodes[item.bodynode_id2].name == "r-foot":
                    c+=1

                if self.robot_skeleton.bodynodes[item.bodynode_id2].name == "l-foot":
                    c+=1






        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 200).all() and# (abs(L_angle - self.foot_angles[self.count]) < 10) and (abs(R_angle - self.foot_angles[self.count]) < 10) and
                (height > -0.70) and (height < 0.40) and (abs(self.robot_skeleton.q[4]) < 0.30) and (abs(self.robot_skeleton.q[5]) < 0.50) and (self.robot_skeleton.q[3] > -0.4) and (self.robot_skeleton.q[3] < 0.3))

        flag = 0

        if done:
            reward = 0.
            flag = 1


        ob = self._get_obs()
        #print("here")
        if self.dumpStates:
            with open("states_from_net.txt","ab") as fp:
                np.savetxt(fp,np.array([ob]),fmt='%1.5f')
        if self.trainRelay:
            ac,vpred = self.pol.act(False,ob)
            #print("vpred",vpred)
            if vpred > 4000:
                print("yipeee",vpred)
                reward =  10*vpred#/100
                done = True

        if head_flag:
            reward = 0.
            done = True

        self.prevdq = np.copy(self.robot_skeleton.dq)


        self.t += self.dt


        reward_breakup = {'r':np.array([flag])}#,-total_force_mag/1000., -1e-2*np.sum(self.robot_skeleton.dq[[6,12,9,15,10,16]]), 10*self.robot_skeleton.dq[2], 10*self.robot_skeleton.dq[1],flag])}#{'r':np.array([right_foot_reward])}#
        if self.dumpRewards:
            with open("reward_terms.txt","ab") as fp:
                np.savetxt(fp,np.array([[root_trans_term,root_trans_vel,joint_term,orient_term,joint_vel_term,0.1*com_height_reward,flag]]),fmt="%1.5f")

            with open("reward.txt","ab") as fp:
                np.savetxt(fp,np.array([[reward]]),fmt='%1.5f')



        self.prev_a = a
        ob = self._get_obs()
        joint_every_diff = np.sum(np.square(self.WalkPositions[:,6:] - self.robot_skeleton.q[6:]),axis=1)
        min_error = np.argmin(joint_every_diff)
        #print(self.count)
        self.count+=1
        if self.count>= self.WalkPositions.shape[0]-1:#449
            done = True

        self.dart_world.set_text.append(str(done))
        #done = False
        return ob, reward, done, {"end_effector_reward":end_effector_reward,"joint_vel_term":joint_vel_term,"joint_term":quat_term,
        "com_reward":com_reward}

    def _get_obs(self):


        phi = np.array([self.count/self.WalkPositions.shape[0]])
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

    def reset_model(self):
        if self.firstPass and self.trainRelay:
            print("yes")
            sess = tf.get_default_session()
            for item in tf.global_variables():
                name = str(item.name)
                if 'node_0' in name:
                    #print(name)
                    obj = item.assign(self.par[item.name])
                    sess.run(obj)
            self.firstPass = False
        self.dart_world.reset()

        phases = [0,1,2]
        phase_index = np.random.choice(phases,1,p=[0.5,0.25,0.25])

        phase_index = 0



        rand_start = np.random.randint(low=1,high=self.WalkPositions.shape[0],size=1)#self.WalkPositions.shape[0]

        qpos = self.WalkPositions[rand_start[0],:].reshape(29,) +self.np_random.uniform(low=-0.0050, high=.0050, size=self.robot_skeleton.ndofs)

        self.count =rand_start[0]
        qvel = self.WalkVelocities[rand_start[0],:].reshape(29,) + self.np_random.uniform(low=-0.0050, high=.0050, size=self.robot_skeleton.ndofs)

        self.set_state(qpos, qvel)
        self.t = 0

        self.count2 = 0
        self.impactCount = 0
        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = 5.0
            self._get_viewer().scene.tb.trans[2] = -7.5
            self._get_viewer().scene.tb.trans[1] = 0.0
            #-10.0


def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)
