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

# TODO Move these things to a proper path!!
from gym.envs.dart.euclideanSpace import *
from gym.envs.dart.quaternions import *

from enum import Enum
import pydart2 as pydart
import argparse
from amc import AMC
from asf_skeleton import ASF_Skeleton
from joint import expand_angle
from transformations import *

class StateMode(Enum):

    GEN_EULER = 0
    GEN_QUAT = 1
    GEN_AXIS = 2

    MIX_EULER = 3
    MIX_QUAT = 4
    MIX_AXIS = 5

def dart_dof_data(amc_frame, skel_dofs, asf):

    # TODO Find a way of extracting information other than
    # relying on the underlying amc file
    # dof_data is dof_name -> (range of indices, order )
    dof_data = {}
    for dof_name, _ in amc_frame:
        joint = asf.name2joint[dof_name]
        axes_str = joint.dofs
        indices = [i for i, dof in enumerate(skel_dofs)
                   if dof.name.startswith(dof_name)]
        dof_data[dof_name] = (indices, axes_str)

    # Change the root just to make things a bit easier
    del dof_data["root"]
    dof_data["root_theta"] = ([3, 4, 5], "xyz")

    return dof_data

class DartDeepMimic(dart_env.DartEnv):

    def __init__(self, control_skeleton_path, asf_path,
                 reference_motion_path,
                 statemode = StateMode.GEN_EULER,
                 actionmode = StateMode.GEN_EULER,
                 visualize=True, frame_skip=16, dt=.005,
                 obs_type="parameter",
                 action_type="continuous", screen_width=80,
                 screen_height=45):


        ###########################################################
        # Extract dof info so that states can be converted easily #
        ###########################################################

        world = pydart.World(dt, control_skeleton_path)
        asf = ASF_Skeleton(asf_path)

        self.ref_skel = world.skeletons[1]
        self.mocap_data = AMC(reference_motion_path)
        self.dart_dof_data = dart_dof_data(self.mocap_data.frames[0],
                                           self.ref_skel.dofs, asf)
        # Setting control skel to ref skel is just a workaround:
        # it's set to its correct value later on
        self.control_skel = self.ref_skel

        ######################################################
        # Set the _get_obs function based on the chosen mode #
        ######################################################

        if statemode == StateMode.GEN_EULER.value:
            self._get_obs = self.gen_as_euler
        elif statemode == StateMode.GEN_QUAT.value:
            self._get_obs = self.gen_as_quat
        elif statemode == StateMode.GEN_AXIS.value:
            self._get_obs = self.gen_as_axisangle
        else:
            raise RuntimeError("Unrecognized or unimpletmented code: "
                               + str(statemode))

        dart_env.DartEnv.__init__(self, [control_skeleton_path], frame_skip,
                                  len(self._get_obs()),
                                  control_bounds, dt, obs_type,
                                  action_type, visualize, not visualize)

        self.control_skel = self.dart_world.skeletons[1]

        # self.control_bounds = np.array([10*np.ones(32,), -10*np.ones(32,)])

        # obs_dim = 58 #+3#for phase
        # ### POINTS ON THE FOOT PLANE
        # self.P = np.array([0.,0.,-0.0525])
        # self.Q = np.array([-0.05,0.,-0.05])
        # self.R = np.array([-0.05,0.,0.])

        # self.duplicate = False
        # self.switch = -1
        # self.impactCount = 0
        # self.storeState = False
        # self.init_q = np.zeros(29,)
        # self.init_dq = np.zeros(29,)
        # self.balance_PID = False
        # self.swingFoot = 'Right'

        # self.count = 0
        # self.count2 = 1
        # self.count_left = self.count_right = 0
        # self.prev_a = np.zeros(23,)
        # self.init_count = 0
        # self.balance = False
        # self.trainRelay = False
        # self.firstPass =  False
        # self.qpos_node0 = np.zeros(29,)
        # self.qpos_node1 = np.zeros(29,)
        # self.qpos_node2 = np.zeros(29,)
        # self.qpos_node3 = np.zeros(29,)
        # #prefix = '../../Balance_getup/'
        # prefix = './'

        #self.WalkPositions[:,1]+=0.013
        #self.WalkVelocities/=1.0
        #self.rarm_endeffector = self.rarm_endeffector[1:]
        #self.larm_endeffector = self.larm_endeffector[1:]
        #self.rfoot_endeffector = self.rfoot_endeffector[1:]
        #self.lfoot_endeffector = self.lfoot_endeffector[1:]
        #self.com = self.com[1:]

        # high = np.inf*np.ones(obs_dim)
        # low = -high
        # observation_space = spaces.Box(low, high)
        # action_space = spaces.Box(10*np.ones(31,), -10*np.ones(31,))

        # if self.trainRelay:
        #     with open("Human_node0_tuned.pkl","rb") as fp:#../../Balance_getup/
        #         self.par = pickle.load(fp)
        #     self.pol = MlpPolicy("Human_node_0",observation_space,action_space,hid_size=128,num_hid_layers=2)
        #     #tf.global_variables_initializer()
        #     sess = tf.get_default_session()
        #     for item in self.pol.get_variables():
        #         name = str(item.name)
        #         if 'node_0' in name:
        #             obj = item.assign(self.par[item.name])
        #             sess.run(obj)
        '''
        with open('positions_new_walk_rightswing.txt','rb') as fp:
            self.ref_trajectory_right = np.loadtxt(fp)

        #self.ref_trajectory = self.ref_trajectory[2:,:]
        with open('vel_new_walk_rightswing.txt','rb') as fp:
            self.ref_vel_right = np.loadtxt(fp)

        #self.ref_vel[:,3]-=np.pi/20
        #self.ref_vel = self.ref_vel[2:,:]
        with open("acc_new_walk_rightswing.txt","rb") as fp:
            self.ref_acc_right = np.loadtxt(fp)


        with open('positions_new_walk_leftswing.txt','rb') as fp:
            self.ref_trajectory_left = np.loadtxt(fp)

        #self.ref_trajectory = self.ref_trajectory[2:,:]
        with open('vel_new_walk_leftswing.txt','rb') as fp:
            self.ref_vel_left = np.loadtxt(fp)

        #self.ref_vel[:,3]-=np.pi/20
        #self.ref_vel = self.ref_vel[2:,:]
        with open("acc_new_walk_leftswing.txt","rb") as fp:
            self.ref_acc_left = np.loadtxt(fp)


        #self.ref_acc = self.ref_acc[2:,:]
        with open("foot_angles.txt","rb") as fp:
            self.foot_angles = np.loadtxt(fp)
        self.foot_angles = self.foot_angles[2:]
        #b, a = signal.butter(4, 0.525)
        '''
        #self.ref_acc = signal.filtfilt(b,a,self.ref_acc)

        # self.prevdq = np.zeros(29,)
        # self.tau = np.zeros(29,)
        # self.ndofs = 29
        # self.target = np.zeros(self.ndofs,)
        # self.init = np.zeros(self.ndofs,)
        # self.edot = np.zeros(self.ndofs,)
        # self.preverror = np.zeros(self.ndofs,)
        # self.stance = None
        # self.target_vel = 1.0
        # for i in range(6, self.ndofs):
        #     self.preverror[i] = (self.init[i] - self.target[i])
        # self.t = 0

        # dart_env.DartEnv.__init__(self, ['kima/kima_human_box.skel'],16, obs_dim, self.control_bounds, disableViewer=False)

        # self.robot_skeleton.set_self_collision_check(True)

        # utils.EzPickle.__init__(self)

    def expand_state(self, generalized_q):
        """
        Given an obs (q or dq, don't really matter), turn it into
        fully-expanded euler angles (w/ three params each), xyz order; returns
        dictionary

        Return tuple like root_pos, {dict of all angles}
        """

        root_translation = generalized_q[0:3]
        expanded_angles = {}
        for dof_name in self.dart_dof_data:
            indices, order = self.dart_dof_data[dof_name]
            fi = indices[0]
            li = indices[-1]
            expanded_angles[dof_name] = expand_angle(generalized_q[fi:li],
                                                     order)
        return root_translation, expanded_angles

    def gen_state_components(self):

        ############################################################
        # TODO Is unfunctional, relies on control_skel rather than #
        # a parameter                                              #
        ############################################################

        pos, angles_dict = self.expand_state(self.control_skel.q)
        dpos, dangles_dict = self.expand_state(self.control_skel.dq)

        angles = np.array(list(angles_dict.values()))
        dangles = np.array(list(dangles_dict.values()))

        gen_pos = np.concatenate([pos, dpos])
        gen_angles = np.concatenate([angles, dangles])

        return gen_pos, gen_angles

    def gen_as_transform(self, tform):
        pos, angles = self.gen_state_components()
        transformed_angles = [tform(angle) for angle in angles]
        flat_angles = np.array(transformed_angles).flatten()
        return np.concatenate([pos, flat_angles])

    def gen_as_quat(self):
        return self.gen_as_transform(lambda x: quaternion_from_euler(*x, axes="rxyz"))

    def gen_as_euler(self):
        return self.gen_as_transform(lambda x: x)

    def gen_as_axisangle(self):
        return self.gen_as_transform(lambda x: axisangle_from_euler(*x, axes="rxyz"))

    # def _get_obs(self):
    #     raise NotImplementedError()

    def transformActions(self,actions):
        raise NotImplementedError()
        # joint_targets = np.zeros(23,)
        # # Left thigh
        # lthigh = actions[:4]
        # euler_lthigh = angle_axis2euler(theta=lthigh[0],vector=lthigh[1:])
        # joint_targets[0] = euler_lthigh[2]
        # joint_targets[1] = euler_lthigh[1]
        # joint_targets[2] = euler_lthigh[0]
        # ###### Left Knee
        # joint_targets[3] = actions[4]
        # ### left foot
        # lfoot = actions[5:9]
        # euler_lfoot = angle_axis2euler(theta=lfoot[0],vector=lfoot[1:])
        # joint_targets[4] = euler_lfoot[2]
        # joint_targets[5] = euler_lfoot[0]

        # # right thigh
        # rthigh = actions[9:13]
        # euler_rthigh = angle_axis2euler(theta=rthigh[0],vector=rthigh[1:])
        # joint_targets[6] = euler_rthigh[2]
        # joint_targets[7] = euler_rthigh[1]
        # joint_targets[8] = euler_rthigh[0]
        # ###### right Knee
        # joint_targets[9] = actions[13]
        # ### right foot
        # rfoot = actions[14:18]
        # euler_rfoot = angle_axis2euler(theta=rfoot[0],vector=rfoot[1:])
        # joint_targets[10] = euler_rfoot[2]
        # joint_targets[11] = euler_rfoot[0]

        # ###thorax

        # thorax = actions[18:22]
        # euler_thorax = angle_axis2euler(theta=thorax[0],vector=thorax[1:])
        # joint_targets[12] = euler_thorax[2]
        # joint_targets[13] = euler_thorax[1]
        # joint_targets[14] = euler_thorax[0]

        # #### l upper arm
        # l_arm = actions[22:26]
        # #print("LARM",actions.shape)
        # euler_larm = angle_axis2euler(theta=l_arm[0],vector=l_arm[1:])
        # joint_targets[15] = euler_larm[2]
        # joint_targets[16] = euler_larm[1]
        # joint_targets[17] = euler_larm[0]

        # ## l elbow

        # joint_targets[18] = actions[25]

        # ## r upper arm
        # r_arm = actions[27:31]
        # euler_rarm = angle_axis2euler(theta=r_arm[0],vector=r_arm[1:])
        # joint_targets[19] = euler_rarm[2]
        # joint_targets[20] = euler_rarm[1]
        # joint_targets[21] = euler_rarm[0]

        # ###r elbow

        # joint_targets[22] = actions[30]

        # return joint_targets

    def ComputeReward(self,):
        raise NotImplementedError()
        # quaternion_difference = []
        # #### lthigh
        # lthigh_euler = self.robot_skeleton.q[6:9]
        # lthigh_mocap = self.WalkPositions[self.count,6:9]
        # quat_lthigh = euler2quat(z=lthigh_euler[2],y=lthigh_euler[1],x=lthigh_euler[0])
        # quat_lthigh_mocap = euler2quat(z=lthigh_mocap[2],y=lthigh_mocap[1],x=lthigh_mocap[0])
        # lthigh_diff = mult(inverse(quat_lthigh_mocap),quat_lthigh)
        # scalar_lthigh = 2*np.arccos(lthigh_diff[0])
        # quaternion_difference.append(scalar_lthigh)
        # #print("scaler",scalar_lthigh)
        # ##### lknee
        # lknee_euler = self.robot_skeleton.q[9]
        # lknee_mocap = self.WalkPositions[self.count,9]
        # quat_lknee = euler2quat(z=0.,y=0.,x=lknee_euler)
        # quat_lknee_mocap = euler2quat(z=0.,y=0.,x=lknee_mocap)
        # lknee_diff = mult(inverse(quat_lknee_mocap),quat_lknee)
        # scalar_lknee = 2*np.arccos(lknee_diff[0])
        # quaternion_difference.append(scalar_lknee)
        # #### lfoot
        # lfoot_euler = self.robot_skeleton.q[10:12]
        # lfoot_mocap = self.WalkPositions[self.count,10:12]
        # quat_lfoot = euler2quat(z=lfoot_euler[1],y=0.,x=lfoot_euler[0])
        # quat_lfoot_mocap = euler2quat(z=lfoot_mocap[1],y=0.,x=lfoot_mocap[0])
        # lfoot_diff = mult(inverse(quat_lfoot_mocap),quat_lfoot)
        # scalar_lfoot = 2*np.arccos(lfoot_diff[0])
        # quaternion_difference.append(scalar_lfoot)
        # #### rthigh
        # rthigh_euler = self.robot_skeleton.q[12:15]
        # rthigh_mocap = self.WalkPositions[self.count,12:15]
        # quat_rthigh = euler2quat(z=rthigh_euler[2],y=rthigh_euler[1],x=rthigh_euler[0])
        # quat_rthigh_mocap = euler2quat(z=rthigh_mocap[2],y=rthigh_mocap[1],x=rthigh_mocap[0])
        # rthigh_diff = mult(inverse(quat_rthigh_mocap),quat_rthigh)
        # scalar_rthigh = 2*np.arccos(rthigh_diff[0])
        # quaternion_difference.append(scalar_rthigh)
        # #print("scaler",scalar_lthigh)
        # ##### rknee
        # rknee_euler = self.robot_skeleton.q[15]
        # rknee_mocap = self.WalkPositions[self.count,15]
        # quat_rknee = euler2quat(z=0.,y=0.,x=rknee_euler)
        # quat_rknee_mocap = euler2quat(z=0.,y=0.,x=rknee_mocap)
        # rknee_diff = mult(inverse(quat_rknee_mocap),quat_rknee)
        # scalar_rknee = 2*np.arccos(rknee_diff[0])
        # quaternion_difference.append(scalar_rknee)
        # #### rfoot
        # rfoot_euler = self.robot_skeleton.q[16:18]
        # rfoot_mocap = self.WalkPositions[self.count,16:18]
        # quat_rfoot = euler2quat(z=rfoot_euler[1],y=0.,x=rfoot_euler[0])
        # quat_rfoot_mocap = euler2quat(z=rfoot_mocap[1],y=0.,x=rfoot_mocap[0])
        # rfoot_diff = mult(inverse(quat_rfoot_mocap),quat_rfoot)
        # scalar_rfoot = 2*np.arccos(rfoot_diff[0])
        # quaternion_difference.append(scalar_rfoot)

        # scalar_thoraxx = self.robot_skeleton.q[18] - self.WalkPositions[self.count,18]
        # quaternion_difference.append(scalar_thoraxx)
        # scalar_thoraxy = self.robot_skeleton.q[19] - self.WalkPositions[self.count,19]
        # quaternion_difference.append(scalar_thoraxy)
        # scalar_thoraxz = self.robot_skeleton.q[20] - self.WalkPositions[self.count,20]
        # quaternion_difference.append(scalar_thoraxz)
        # #### l upper arm
        # larm_euler = self.robot_skeleton.q[21:24]
        # larm_mocap = self.WalkPositions[self.count,21:24]
        # quat_larm = euler2quat(z=larm_euler[2],y=larm_euler[1],x=larm_euler[0])
        # quat_larm_mocap = euler2quat(z=larm_mocap[2],y=larm_mocap[1],x=larm_mocap[0])
        # larm_diff = mult(inverse(quat_larm_mocap),quat_larm)
        # scalar_larm = 2*np.arccos(larm_diff[0])
        # quaternion_difference.append(scalar_larm)
        # #print("scaler",scalar_lthigh)
        # ##### l elbow
        # lelbow_euler = self.robot_skeleton.q[24]
        # lelbow_mocap = self.WalkPositions[self.count,24]
        # quat_lelbow = euler2quat(z=0.,y=0.,x=lelbow_euler)
        # quat_lelbow_mocap = euler2quat(z=0.,y=0.,x=lelbow_mocap)
        # lelbow_diff = mult(inverse(quat_lelbow_mocap),quat_lelbow)
        # scalar_lelbow = 2*np.arccos(lelbow_diff[0])
        # quaternion_difference.append(scalar_lelbow)
        # #### r upper arm
        # rarm_euler = self.robot_skeleton.q[25:28]
        # rarm_mocap = self.WalkPositions[self.count,25:28]
        # quat_rarm = euler2quat(z=rarm_euler[2],y=rarm_euler[1],x=rarm_euler[0])
        # quat_rarm_mocap = euler2quat(z=rarm_mocap[2],y=rarm_mocap[1],x=rarm_mocap[0])
        # rarm_diff = mult(inverse(quat_rarm_mocap),quat_rarm)
        # scalar_rarm = 2*np.arccos(rarm_diff[0])
        # quaternion_difference.append(scalar_rarm)
        # #print("scaler",scalar_lthigh)
        # ##### r elbow
        # relbow_euler = self.robot_skeleton.q[28]
        # relbow_mocap = self.WalkPositions[self.count,28]
        # quat_relbow = euler2quat(z=0.,y=0.,x=relbow_euler)
        # quat_relbow_mocap = euler2quat(z=0.,y=0.,x=relbow_mocap)
        # relbow_diff = mult(inverse(quat_relbow_mocap),quat_relbow)
        # scalar_relbow = 2*np.arccos(relbow_diff[0])
        # quaternion_difference.append(scalar_relbow)

        # quat_reward = np.exp(-2*np.sum(np.square(quaternion_difference)))
        # #print("reward",quat_reward)

        # return quat_reward


    def advance(self, a):
        raise NotImplementedError()
        # clamped_control = np.array(a)

        # self.tau = np.zeros(self.robot_skeleton.ndofs)
        # trans = np.zeros(6,)


        # self.target[6:]=  self.transformActions(clamped_control)# + self.WalkPositions[self.count,6:] #*self.action_scale# + self.ref_trajectory_right[self.count_right,6:]# +

        # self.target[[6,9,10]] = self.target[[12,15,16]]
        # actions = np.zeros(29,)
        # actions[2] = 5
        # actions[6:] = copy.deepcopy(self.target[6:])
        # self.action_skel.set_positions(actions)
        # self.action_skel.set_velocities(np.zeros(29,))

        # for i in range(4):
        #     self.tau[6:] = self.PID()

        #     dupq = copy.deepcopy(self.WalkPositions[self.count,:])
        #     dupq[0] = 0.90
        #     dupq[2] = 0.5

        #     self.dupSkel.set_positions(dupq)

        #     dupdq = np.zeros(29,)
        #     dupdq = copy.deepcopy(self.WalkVelocities[self.count,:])

        #     self.dupSkel.set_velocities(dupdq)

        #     if self.dumpTorques:
        #         with open("torques.txt","ab") as fp:
        #             np.savetxt(fp,np.array([self.tau]),fmt='%1.5f')

        #     if self.dumpActions:
        #         with open("targets_from_net.txt",'ab') as fp:
        #             np.savetxt(fp,np.array([[self.target[6],self.robot_skeleton.q[6]]]),fmt='%1.5f')


        #     self.robot_skeleton.set_forces(self.tau)
        #     #print("torques",self.tau[22])
        #     self.dart_world.step()

        #self.do_simulation(self.tau, self.frame_skip)

    def ClampTorques(self,torques):
        raise NotImplementedError()
        # torqueLimits = np.array([150.0*5,80.*3,80.*3,100.*5,80.*5,60.,150.0*5,80.*3,80.*3,100.*5,80.*5,60.,150.*5,150.*5,150.*5,10.,5.,5.,5.,10.,5.,5,5.])*2
        # #print("tau",torqueLimits[4])
        # for i in range(6,self.ndofs):
        #     if torques[i] > torqueLimits[i-6]:
        #         torques[i] = torqueLimits[i-6]
        #     if torques[i] < -torqueLimits[i-6]:
        #         torques[i] = -torqueLimits[i-6]

        # return torques

    def PID(self):
        raise NotImplementedError()
        #print("#########################################################################3")
        # self.kp = np.array([250]*23)#350
        # self.kd = np.array([0.005]*23)

        # self.kp[0] = 600+25 #1000#
        # self.kp[3] = 225+25#0#450 #
        # self.kp[9] = 225+25
        # self.kp[10] = 200
        # self.kp[16] = 200
        # self.kp[[1,2]] = 150
        # self.kp[[7,8]] = 150
        # self.kp[6] =600+25
        # #self.kd[[4,510,11]] = 0.005
        # self.kp[15:] = 155
        # self.kd[15:]= 0.05

        # self.kp = [item/2  for item in self.kp]
        # self.kd = [item/2  for item in self.kd]

        # q = self.robot_skeleton.q
        # qdot = self.robot_skeleton.dq
        # tau = np.zeros((self.ndofs,))
        # for i in range(6, self.ndofs):
        #     #print(q.shape)
        #     self.edot[i] = ((q[i] - self.target[i]) -
        #         self.preverror[i]) / self.dt
        #     tau[i] = -self.kp[i - 6] * \
        #         (q[i] - self.target[i]) - \
        #         self.kd[i - 6] *qdot[i]
        #     self.preverror[i] = (q[i] - self.target[i])

        # torqs = self.ClampTorques(tau)

        # return torqs[6:]
        #return tau[6:]

    def _step(self, a):

        raise NotImplementedError()
        # self.dart_world.set_text = []
        # self.dart_world.y_scale = np.clip(a[6],-2,2)
        # self.dart_world.plot = False
        # count_str = "count :"+str(self.count)
        # a_from_net = "a[6] : %f and a[12] : %f"%(a[16],a[20])
        # self.dart_world.set_text.append(a_from_net)
        # self.dart_world.set_text.append(count_str)
        # posbefore = self.robot_skeleton.bodynodes[0].com()[0]


        # if self.duplicate:
        #     dupq = self.robot_skeleton.q
        #     dupq[0] = 1.0
        #     self.dupSkel.set_positions(dupq)
        #     self.dupskel.set_velocities(np.zeros(self.robot_skeleton.q.shape[0],))


        # self.advance(a)
        # if self.dumpActions:
        #     with open("a_from_net.txt","ab") as fp:
        #         np.savetxt(fp,np.array([a]),fmt='%1.5f')

        # #with open("states_PID_jumpPol13.txt","ab") as fp:
        # #    np.savetxt(fp,np.array([self.robot_skeleton.q]),fmt='%1.5f')
        # #with open("velocities_PID_jumpPol14.txt","ab") as fp:
        # #    np.savetxt(fp,np.array([self.robot_skeleton.dq]),fmt='%1.5f')
        # #with open("torques_PIDPol14.txt","ab") as fp:
        # #   np.savetxt(fp,np.array([self.tau]),fmt='%1.5f')

        # #print("torques",self.tau[[6,12]])
        # point_rarm = [0.,-0.60,-0.15]
        # point_larm = [0.,-0.60,-0.15]
        # point_rfoot = [0.,0.,-0.20]
        # point_lfoot = [0.,0.,-0.20]

        # #with open("PID_rarm.txt","ab") as fp:
        # #    np.savetxt(fp,np.array([self.robot_skeleton.bodynodes[16].to_world(point_rarm)]),fmt='%1.5f')

        # #with open("PID_larm.txt","ab") as fp:
        # #    np.savetxt(fp,np.array([self.robot_skeleton.bodynodes[13].to_world(point_larm)]),fmt='%1.5f')

        # #with open("PID_rfoot.txt","ab") as fp:
        # #    np.savetxt(fp,np.array([self.robot_skeleton.bodynodes[7].to_world(point_rfoot)]),fmt='%1.5f')

        # #with open("PID_lfoot.txt","ab") as fp:
        # #    np.savetxt(fp,np.array([self.robot_skeleton.bodynodes[4].to_world(point_lfoot)]),fmt='%1.5f')

        # global_rarm = self.robot_skeleton.bodynodes[16].to_world(point_rarm)

        # global_larm = self.robot_skeleton.bodynodes[13].to_world(point_larm)
        # global_lfoot = self.robot_skeleton.bodynodes[4].to_world(point_lfoot)
        # global_rfoot = self.robot_skeleton.bodynodes[7].to_world(point_rfoot)

        # global_rarmdup = self.dupSkel.bodynodes[16].to_world(point_rarm)
        # global_larmdup = self.dupSkel.bodynodes[13].to_world(point_larm)
        # global_lfootdup = self.dupSkel.bodynodes[4].to_world(point_lfoot)
        # global_rfootdup = self.dupSkel.bodynodes[7].to_world(point_rfoot)

        # self.dart_world.contact_point = []
        # self.dart_world.contact_color = 'red'
        # self.dart_world.contact_point.append(global_rarm)
        # self.dart_world.contact_point.append(global_larm)
        # self.dart_world.contact_point.append(global_rfoot)
        # self.dart_world.contact_point.append(global_lfoot)
        # self.dart_world.contact_color = 'green'
        # self.dart_world.contact_point.append(global_rarmdup)
        # self.dart_world.contact_point.append(global_larmdup)
        # self.dart_world.contact_point.append(global_rfootdup)
        # self.dart_world.contact_point.append(global_lfootdup)


        # #print(self.swingFoot)
        # posafter = self.robot_skeleton.bodynodes[0].com()[0]
        # height = self.robot_skeleton.bodynodes[0].com()[1]
        # side_deviation = self.robot_skeleton.bodynodes[0].com()[2]

        # upward = np.array([0, 1, 0])
        # upward_world = self.robot_skeleton.bodynode('head').to_world(
        #     np.array([0, 1, 0])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        # upward_world /= np.linalg.norm(upward_world)
        # ang_cos_uwd = np.dot(upward, upward_world)
        # ang_cos_uwd = np.arccos(ang_cos_uwd)

        # forward = np.array([1, 0, 0])
        # forward_world = self.robot_skeleton.bodynode('head').to_world(
        #     np.array([1, 0, 0])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        # forward_world /= np.linalg.norm(forward_world)
        # ang_cos_fwd = np.dot(forward, forward_world)
        # ang_cos_fwd = np.arccos(ang_cos_fwd)

        # lateral = np.array([0, 0, 1])
        # lateral_world = self.robot_skeleton.bodynode('head').to_world(
        #     np.array([0, 0, 1])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        # lateral_world /= np.linalg.norm(lateral_world)
        # ang_cos_ltl = np.dot(lateral, lateral_world)
        # ang_cos_ltl = np.arccos(ang_cos_ltl)

        # contacts = self.dart_world.collision_result.contacts

        # if contacts == []:
        #     self.switch = 0

        # if contacts != []:
        #     self.switch = 1


        # self.ComputeReward()

        # total_force_mag = 0
        # for contact in contacts:


        #     force = np.sum(contact.force[[1,2]])



        #     total_force_mag += force
        #     data = np.zeros(11,)
        #     data[:10] = contact.state
        #     data[10] = self.count


        # if self.dumpCOM:
        #     with open("COM_fromPolicy_walk.txt","ab") as fp:
        #         np.savetxt(fp,np.asarray(self.robot_skeleton.q[:3]),fmt="%1.5f")

        #     with open("Joint_fromPolicy_walk.txt","ab") as fp:
        #         np.savetxt(fp,np.asarray(self.robot_skeleton.q),fmt="%1.5f")



        # alive_bonus = 4
        # vel = (posafter - posbefore) / self.dt
        # #print("a shape",a[-1])
        # action_pen = np.sqrt(np.square(a[:23]).sum())

        # reward = 0


        # W_joint = 2.
        # W_joint_vel = 2.
        # W_trans = 2.
        # W_orient = 1.

        # #W_theta = 0.5
        # W_joint_vel = 1.
        # W_trans_vel = 0.5
        # W_orient_vel = 0.5
        # W_balance = 5.0

        # Joint_weights = np.ones(23,)#
        # Joint_weights[[0,3,6,9,16,20,10,16]] = 10

        # Weight_matrix = np.diag(Joint_weights)
        # Weight_matrix_1 = np.diag(np.ones(4,))

        # veldq = np.copy(self.robot_skeleton.dq)
        # #print("ve",veldq)
        # acc = (veldq - self.prevdq)/0.008


        # com_height = np.square(0 - self.robot_skeleton.bodynodes[0].com()[1])
        # com_height_reward = 10*np.exp(-5*com_height)
        # right_foot = np.square(0 - self.robot_skeleton.bodynodes[7].com()[1])
        # right_foot_reward = 10*np.exp(-5*right_foot)

        # #print("rew",right_foot_reward)


        # done = False


        # rarm_term = np.sum(np.square(self.rarm_endeffector[self.count,:] - global_rarm))
        # larm_term = np.sum(np.square(self.larm_endeffector[self.count,:] - global_larm))
        # rfoot_term = np.sum(np.square(self.rfoot_endeffector[self.count,:] - global_rfoot))
        # lfoot_term = np.sum(np.square(self.lfoot_endeffector[self.count,:] - global_lfoot))

        # end_effector_reward = np.exp(-40*(rarm_term+larm_term+rfoot_term+lfoot_term))
        # com_reward = np.exp(-40*np.sum(np.square(self.com[self.count,:] - self.robot_skeleton.bodynodes[0].com())))

        # s = self.state_vector()







        # joint_diff = self.WalkPositions[self.count,6:] - self.robot_skeleton.q[6:]#hmm[[6,9,12,15,22,26,10,16]] - self.robot_skeleton.q[[6,9,12,15,22,26,10,16]]
        # #joint_diff_unimp = hmm[[7,8,13,14]] - self.robot_skeleton.q[[7,8,13,14]]
        # joint_pen = np.sum(joint_diff.T*Weight_matrix*joint_diff)
        # #joint_pen_unimp = np.sum(joint_diff_unimp.T*Weight_matrix_1*joint_diff_unimp)

        # vel_diff = self.WalkVelocities[self.count,6:] - self.robot_skeleton.dq[6:]

        # vel_pen = np.sum(vel_diff.T*Weight_matrix*vel_diff)

        # node1_trans = np.array([0,-0.25,0])
        # node1_root_orient = np.array([-np.pi/5,0,0])
        # node0_trans = self.qpos_node0[:3]

        # trans_pen = np.sum(np.square(node0_trans[:3] - self.robot_skeleton.q[:3]))
        # trans_vel_pen = np.sum(np.square(np.zeros(3,) - self.robot_skeleton.dq[:3]))
        # root_orient_pen = np.sum(np.square(np.zeros(3,) - self.robot_skeleton.q[3:6]))
        # root_orient_vel = np.sum(np.square(self.init_dq[3:6] - self.robot_skeleton.dq[3:6]))

        # #print("com",self.robot_skeleton.bodynodes[0].com())
        # ##node1
        # root_node_com = np.array([0,-0.10,0]) #
        # trans_pen = np.sum(np.square(root_node_com - self.robot_skeleton.bodynodes[0].com()))
        # trans_vel_pen = np.sum(np.square(self.robot_skeleton.dq[:3]))
        # root_orient_pen = np.sum(np.square(np.zeros(3,) - self.robot_skeleton.q[3:6]))
        # root_orient_vel = np.sum(np.square(self.init_dq[3:6] - self.robot_skeleton.dq[3:6]))
        # #orient_vel = np.copy(self.ref_vel[int(self.count/10),3:6])
        # #trans_vel = np.copy(self.ref_trajectory[int(self.count/10),[2,1,0]])
        # #trans_vel_pen = np.sum(np.square(trans_vel - self.robot_skeleton.dq[3:6]))
        # #joint_vel_pen = np.sum(np.square(self.ref_vel[int(self.count/10),6:18] - self.robot_skeleton.dq[6:18]))
        # root_trans_term = 10/(1+ 100*trans_pen)#np.asarray(W_trans*np.exp(-10*trans_pen))
        # #
        # root_trans_vel = 100/(1+ 100*trans_vel_pen)#np.asarray(W_joint*np.exp(-10*trans_vel_pen))
        # #
        # joint_term = 1*np.asarray(np.exp(-2e-1*joint_pen))#np.asarray(W_joint*np.exp(-1e-2*joint_pen)) #100
        # #joint_term_unimp = np.asarray(W_joint*np.exp(-joint_pen_unimp))
        # #
        # joint_vel_term = 1*np.asarray(np.exp(-1e-1*vel_pen))# W_joint_vel*np.exp(-1e-3*vel_pen)
        # #20 for root nod0
        # orient_term = 10*np.asarray(W_orient*np.exp(-10*root_orient_pen))

        # com_height = self.robot_skeleton.bodynodes[0].com()[1]
        # contact_reward = 0.
        # if self.count > 230 and self.count < 300 and contacts == []:
        #     contact_reward = 10.

        # quat_term = self.ComputeReward()
        # reward = 0.1*end_effector_reward + 0.1*joint_vel_term+ 0.25*com_reward+ 1.65*quat_term# + contact_reward#  + joint_term + joint_vel_term #0.1*self.robot_skeleton.bodynodes[0].com()[1] + joint_term + joint_vel_term +
        # eerew_str = "End Effector :"+str(end_effector_reward)
        # self.dart_world.set_text.append(eerew_str)

        # vel_str = "Joint Vel :"+str(joint_vel_term)
        # self.dart_world.set_text.append(vel_str)

        # com_str = "Com  :"+str(com_reward)
        # self.dart_world.set_text.append(com_str)

        # joint_str = "Joint :"+str(quat_term)
        # self.dart_world.set_text.append(joint_str)

        # joint_str = "contact :"+str(contact_reward)
        # self.dart_world.set_text.append(joint_str)

        # lthigh_str = "Left Thigh target:"+str(self.WalkPositions[self.count,6])+" thigh Position :"+str(self.robot_skeleton.q[6])
        # self.dart_world.set_text.append(lthigh_str)
        # rthigh_str = "right Thigh target:"+str(self.WalkPositions[self.count,12])+" thigh Position :"+str(self.robot_skeleton.q[12])
        # self.dart_world.set_text.append(rthigh_str)

        # lthigh_torque = "Left Thigh torque:"+str(self.tau[6])
        # self.dart_world.set_text.append(lthigh_torque)
        # rthigh_torque = "right Thigh torque:"+str(self.tau[12])
        # self.dart_world.set_text.append(rthigh_torque)

        # com_vel = "com_vel:"+str(self.robot_skeleton.q[1])
        # self.dart_world.set_text.append(com_vel)

        # tar_vel = "tar_com_vel:"+str(self.WalkVelocities[self.count,1])
        # self.dart_world.set_text.append(tar_vel)


        # c = 0
        # head_flag = False
        # for item in contacts:
        #     #c = 0
        #     if item.skel_id1 == 0:
        #         if self.robot_skeleton.bodynodes[item.bodynode_id2].name == "head":
        #             #print("Headddddddd")
        #             head_flag = True
        #         if self.robot_skeleton.bodynodes[item.bodynode_id2].name == "l-lowerarm":
        #             c+=1
        #         if self.robot_skeleton.bodynodes[item.bodynode_id2].name == "r-lowerarm":
        #             #print("true")
        #             c+=1
        #         if self.robot_skeleton.bodynodes[item.bodynode_id2].name == "r-foot":
        #             c+=1

        #         if self.robot_skeleton.bodynodes[item.bodynode_id2].name == "l-foot":
        #             c+=1






        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 200).all() and# (abs(L_angle - self.foot_angles[self.count]) < 10) and (abs(R_angle - self.foot_angles[self.count]) < 10) and
        #         (height > -0.7) and  (self.robot_skeleton.q[3] > -0.4) and (self.robot_skeleton.q[3]<0.3) and (abs(self.robot_skeleton.q[4]) < 0.30) and (abs(self.robot_skeleton.q[5]) < 0.30))

        # flag = 0

        # if done:
        #     reward = 0.
        #     flag = 1


        # ob = self._get_obs()

        # if self.dumpStates:
        #     with open("states_from_net.txt","ab") as fp:
        #         np.savetxt(fp,np.array([ob]),fmt='%1.5f')
        # if self.trainRelay:
        #     ac,vpred = self.pol.act(False,ob)
        #     #print("vpred",vpred)
        #     if vpred > 4000:
        #         print("yipeee",vpred)
        #         reward =  10*vpred#/100
        #         done = True

        # if head_flag:
        #     reward = 0.
        #     done = True

        # self.prevdq = np.copy(self.robot_skeleton.dq)


        # self.t += self.dt


        # reward_breakup = {'r':np.array([flag])}#,-total_force_mag/1000., -1e-2*np.sum(self.robot_skeleton.dq[[6,12,9,15,10,16]]), 10*self.robot_skeleton.dq[2], 10*self.robot_skeleton.dq[1],flag])}#{'r':np.array([right_foot_reward])}#
        # if self.dumpRewards:
        #     with open("reward_terms.txt","ab") as fp:
        #         np.savetxt(fp,np.array([[root_trans_term,root_trans_vel,joint_term,orient_term,joint_vel_term,0.1*com_height_reward,flag]]),fmt="%1.5f")

        #     with open("reward.txt","ab") as fp:
        #         np.savetxt(fp,np.array([[reward]]),fmt='%1.5f')



        # self.prev_a = a
        # ob = self._get_obs()
        # joint_every_diff = np.sum(np.square(self.WalkPositions[:,6:] - self.robot_skeleton.q[6:]),axis=1)
        # min_error = np.argmin(joint_every_diff)
        # #print("joint joint_every_diff",min_error)
        # #self.count = min_error
        # self.count+=1
        # if self.count>= 322:#449
        #     done = True

        # self.dart_world.set_text.append(str(done))

        # return ob, reward, done,reward_breakup


    def reset_model(self):
        raise NotImplementedError()
        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = 5.0
            self._get_viewer().scene.tb.trans[2] = -7.5
            self._get_viewer().scene.tb.trans[1] = 0.0
            #-10.0

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Make a DartDeepMimic Environ')
    parser.add_argument('--control-skel-path', required=True,
                        help='Path to the control skeleton')
    parser.add_argument('--asf-path', required=True,
                        help='Path to asf which the skeleton was parsed from')
    parser.add_argument('--ref-motion-path', required=True,
                        help='Path to the reference motion AMC')
    parser.add_argument('--state-mode', default=0, type=int,
                        help="Code for the state representation")
    parser.add_argument('--action-mode', default=0, type=int,
                        help="Code for the action representation")
    parser.add_argument('--visualize', default=True,
                        help="True if you want a window to render to")
    parser.add_argument('--frame-skip', type=int, default=16,
                        help="IDK what this does")
    parser.add_argument('--dt', type=float, default=.002,
                        help="Dart simulation resolution")
    parser.add_argument('--obs-class', default="parameter",
                        help="I have no iea what this does")
    parser.add_argument('--action-class', default="continuous",
                        help="I have no iea what this does")
    parser.add_argument('--window-width', type=int, default=80,
                        help="Window width")
    parser.add_argument('--window-height', type=int, default=45,
                        help="Window height")

    args = parser.parse_args()

    env = DartDeepMimic(args.control_skel_path, args.asf_path,
                        args.ref_motion_path,
                        args.state_mode, args.action_mode, args.visualize,
                        args.frame_skip, args.dt,
                        args.obs_class, args.action_class,
                        args.window_width, args.window_height)
