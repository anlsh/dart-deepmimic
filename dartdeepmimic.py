from gym.envs.dart import dart_env
from math import exp, pi
from numpy.linalg import norm
from transformations import compose_matrix, euler_from_matrix
import argparse
import numpy as np
import pydart2 as pydart
import random
import warnings
from copy import deepcopy
from euclideanSpace import angle_axis2euler, euler2quat
from quaternions import mult, inverse
from math import atan2

# ROOT_KEY isn't customizeable. It should correspond
# to the name of the root node in the amc (which is usually "root")
ROOT_KEY = "root"
GRAVITY_VECTOR = np.array([0, -9.8, 0])

class StateMode:
    """
    Just a convenience enum
    """
    GEN_EULER = 0
    GEN_QUAT = 1
    GEN_AXIS = 2

class ActionMode:
    """
    Another convenience enum
    """
    GEN_EULER = 0
    GEN_QUAT = 1
    GEN_AXIS = 2

    # lengths[code] describes the space needed for an angle of that
    # type. For instance euler is 3 numbers, a quaternion is 4
    lengths = [3, 4, 4]


def pad2length(vector, length):
    padded = np.zeros(length)
    padded[:len(vector)] = deepcopy(vector)
    return padded

def get_metadict(skel):
    """
    Creating a dictionary mapping joint names to list of indices oocupied in
    dof array
    """
    joint_names = [joint.name for joint in skel.joints]
    skel_dofs = skel.dofs

    metadict = {}
    for dof_name in joint_names:
        indices = [i for i, dof in enumerate(skel_dofs)
                   if dof.name.startswith(dof_name)]
        if len(indices) == 0:
            # Weld joints dont have dofs so skip adding them entirely
            continue
        child_body_index = [i for i, body in enumerate(skel.bodynodes)
                            if body.name.startswith(dof_name)][0]
        metadict[dof_name] = (indices, child_body_index)

    return metadict


class DartDeepMimicEnv(dart_env.DartEnv):

    def __init__(self,
                 skel_path,
                 # refmotion_path,
                 # policy_query_frequency,
                 # refmotion_dt,
                 statemode,
                 # actionmode,
                 # p_gain, d_gain,
                 pos_noise, vel_noise,
                 # reward_cutoff,
                 pos_weight, pos_decay,
                 vel_weight, vel_decay,
                 ee_weight, ee_decay,
                 com_weight, com_decay,
                 # max_torque,
                 # max_angle,
                 # delta_actions,
                 # default_damping,
                 # default_spring,
                 # default_friction,
                 # visualize,
                 # simsteps_per_dataframe,
                 # gravity,
                 # self_collide,
                 seed,
    ):

        self.random = random.Random()
        if seed is not None:
            self.random.seed(seed)

        ##############################################
        # Set angle conversion methods appropriately #
        ##############################################

        self.statemode = statemode

        self.angle_to_rep = lambda x: None

        if self.statemode == StateMode.GEN_EULER:
            self.angle_to_rep = lambda x: x
        elif self.statemode == StateMode.GEN_QUAT:
            self.angle_to_rep = lambda theta: euler2quat(z=theta[2],
                                                         y=theta[1],
                                                         x=theta[0])
        elif self.statemode == StateMode.GEN_AXIS:
            raise NotImplementedError()

        self.pos_noise, self.vel_noise = pos_noise, vel_noise
        if self.pos_noise < 0 or self.vel_noise < 0:
            raise RuntimeError("Noise spread should be nonnegative")

        self.framenum = 0

        self.pos_weight, self.pos_decay = pos_weight, pos_decay
        self.vel_weight, self.vel_decay = vel_weight, vel_decay
        self.ee_weight, self.ee_decay = ee_weight, ee_decay
        self.com_weight, self.com_decay = com_weight, com_decay
        if (pos_weight < 0) or (vel_weight < 0) or \
           (ee_weight < 0) or (com_weight) < 0:
            raise RuntimeError("Outer weights should be nonnegative")
        if (pos_decay > 0) or (vel_decay > 0) or \
           (ee_decay > 0) or (com_decay) > 0:
            raise RuntimeError("Decay rates should be nonpositive")


        self.skel_path = skel_path
        ref_skel = pydart.World(.0001, self.skel_path).skeletons[-1]

        self.metadict = get_metadict(ref_skel)

        self._dof_names = [key for key in self.metadict]
        self._dof_names.sort(key=lambda x:self.metadict[x][0][0])
        self._actuated_dof_names = [name for name in self._dof_names
                                    if not name.startswith(ROOT_KEY)]

        #######################################
        # Just set a bunch of self.parameters #
        #######################################

        # self.statemode = statemode
        # self.actionmode = actionmode
        # self.policy_query_frequency = policy_query_frequency
        # TODO Dead variable, re-enable here and in argparse
        # self.refmotion_dt = refmotion_dt
        # self.simsteps_per_dataframe = simsteps_per_dataframe
        # self.max_torque = max_torque
        # self.max_angle = max_angle
        # self.default_damping = default_damping
        # self.default_spring = default_spring
        # self.default_friction = default_friction
        # self.reward_cutoff = reward_cutoff
        # self.gravity = gravity
        # self.self_collide = self_collide
        # self.__visualize = visualize
        # self.delta_actions = delta_actions


        # Set later, simply declaring up front
        # self.robot_skeleton = None
        # self.obs_dim = None
        # self.action_dim = None
        # self.metadict = None
        # self._end_effector_indices = None
        # self.obs_dim = None
        # self.action_dim = None
        # self._dof_names = None
        # self._actuated_dof_names = None

        # self.num_frames = -1
        # self.RefQs = None
        # self.RefDQs = None
        # self.ref_quat_frames = None
        # self.ref_com_frames = None
        # self.ref_ee_frames = None

        ####################################
        # Self.parameters for internal use #
        ####################################

        # TODO Re-enable step resolution calculation
        # self.step_resolution = (1 / self.policy_query_frequency) / self.refmotion_dt
        # self.step_resolution = 4

        # self.angle_from_rep = lambda x: None

        ##############################################
        # Set angle conversion methods appropriately #
        ##############################################

        # Type of angles in state space

        # if self.statemode == StateMode.GEN_EULER:
        #     self.angle_to_rep = lambda x: x

        # elif self.statemode == StateMode.GEN_QUAT:
        #     self.angle_to_rep = lambda x: euler2quat(*(x[::-1]))

        # elif self.statemode == StateMode.GEN_AXIS:
        #     self.angle_to_rep = lambda x: axisangle_from_euler(*(x[::-1]), axes="rxyz")

        # Type of angles in action space

        # if self.actionmode == ActionMode.GEN_EULER:
        #     self.angle_from_rep = lambda x: x

        # elif self.actionmode == ActionMode.GEN_QUAT:
        #     raise NotImplementedError()

        # elif self.actionmode == ActionMode.GEN_AXIS:
        #     self.angle_from_rep = lambda x: angle_axis2euler(theta=x[0],
        #                                                      vector=x[1:])[::-1]

        #################################################
        # Sanity check the values of certain parameters #
        #################################################

        # if not self.gravity:
        #     warnings.warn("Gravity is disabled, be sure you meant to do this!",
                          # RuntimeWarning)
        # if not self.self_collide:
        #     warnings.warn("Self collisions are disabled, be sure you meant"
        #                   + " to do this!", RuntimeWarning)

        # if (self.p_gain < 0) or (self.d_gain < 0):
        #     raise RuntimeError("All PID gains should be positive")


        # if self.step_resolution % 1 != 0:
        #     raise RuntimeError("Refmotion dt doesn't divide query dt")
        # else:
        #     self.step_resolution = int(self.step_resolution)

        #################################################################
        # Extract dof data from skeleton and construct reference frames #
        #################################################################

        # self._end_effector_indices = [i for i, node
        #                                in enumerate(ref_skel.bodynodes)
        #                              if len(node.child_bodynodes) == 0]

        # self.obs_dim = len(self._get_obs(ref_skel))
        # self.action_dim = sum([ActionMode.lengths[self.actionmode]
        #                        if len(self.metadict[name][0]) > 1 else 1
        #                        for name in self._actuated_dof_names])

        # self.RefQs, self.RefDQs, \
            # self.ref_quat_frames, self.ref_com_frames, \
            # self.ref_ee_frames = self.construct_frames(ref_skel,
            #                                            refmotion_path)
        # self.num_frames = len(self.RefQs)

        # TODO Replace the 10 with a max_angle variable
        # action_limits = 10. * np.ones(self.action_dim)
        # action_limits = np.array([action_limits, -action_limits])

        # TODO Hardcoded frame skip, pulled from visak's code
        # TODO bring back my nice keyword :(
        # dart_env.DartEnv.__init__(self,
        #                           model_paths=[self._skeleton_path],
        #                           frame_skip=16,
        #                           observation_size=self.obs_dim,
        #                           action_bounds=action_limits,
        #                           visualize=self.__visualize,
        #                           disableViewer=not self.__visualize)
        # dart_env.DartEnv.__init__(self,
        #                           [self._skeleton_path],
        #                           16,
        #                           self.obs_dim,
        #                           action_limits,
        #                           disableViewer=False)

        #########################################################
        # Set various per joint/body parameters based on inputs #
        #########################################################

        # TODO Re-enable setting joint parameters in here
        # self.dart_world.set_gravity(int(self.gravity) * GRAVITY_VECTOR)
        # self.robot_skeleton.set_self_collision_check(self.self_collide)

        # TODO Re-enable my glorious setting of default values on stuff
        # for joint in self.robot_skeleton.joints[1:]:
        #     if joint.name == ROOT_KEY:
        #         continue
        #     if joint.has_position_limit(0):
        #         joint.set_position_limit_enforced(True)
        #     for index in range(joint.num_dofs()):
        #         joint.set_damping_coefficient(index, self.default_damping)
        #         joint.set_spring_stiffness(index, self.default_spring)

        # for skel in self.dart_world.skeletons:
        #     for body in skel.bodynodes:
        #         body.set_friction_coeff(self.default_friction)

    # def target_angles(self, actuated_angles):
    #     if self.delta_actions:
    #         return self.RefQs[self.framenum][6:] + actuated_angles
    #     else:
    #         return actuated_angles

    def reward(self, skel, framenum):

        diff_pos = self.pos_diff(skel, framenum)
        diff_vel = self.vel_diff(skel, framenum)
        diff_ee = self.ee_diff(skel, framenum)
        diff_com = self.com_diff(skel, framenum)

        return self.pos_weight * np.exp(self.pos_decay * diff_pos) \
            + self.vel_weight * np.exp(self.vel_decay * diff_vel) \
            + self.ee_weight * np.exp(self.ee_decay * diff_ee) \
            + self.com_weight * np.exp(self.com_decay * diff_com)

    def step(self, a):
        return self._step(a)

    def _step(self, nvec):

        # TODO Do I need to clamp anything in this range?
        tau = np.zeros(self.robot_skeleton.ndofs)
        target = np.zeros(self.robot_skeleton.ndofs,)
        target[6:] = self.transformActions(nvec) \
                     + self.MotionPositions[self.framenum,6:]

        # TODO Should be step_resolution instead of 4
        for i in range(4):
            tau[6:] = self.PID(self.robot_skeleton, target)

            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

        R_total = self.reward(self.robot_skeleton, self.framenum)

        s = self.state_vector()
        done = self.should_terminate()

        # TODO Implement proper rude termination
        if done:
            R_total = 0.

        # TODO Implement finiteness check on obs by uncommenting below
        ob = self._get_obs()
        # if not np.isfinite(ob).all():
        #     raise RuntimeError("Ran into an infinite state")

        self.framenum += 1
        if self.framenum >= self.num_frames-1:
            done = True

        return ob, R_total, done, {}

    def get_random_framenum(self, default=None):
        return default if default is not None \
            else self.random.randint(0, self.num_frames - 1)

    # def reset(self, framenum=None, pos_stdv=None, vel_stdv=None):
    #     """
    #     Unfortunately, I have to provide default arguments for pos, vel_stdv
    #     since this is the same method called by the learn function and it
    #     doesn't expect those
    #     """

    #     # I dont actually know what this line of code
    #     self.dart_world.reset()

    #     # TODO Re-enable noise!
    #     self.framenum = self.get_random_framenum(framenum)

    #     self.set_state(self.RefQs[self.framenum],
    #                    self.RefDQs[self.framenum])

    #     return self._get_obs()

    def construct_frames(self, ref_motion_path):
        raise NotImplementedError()

    def _get_ee_positions(self, skel):
        raise NotImplementedError()

    # def _get_obs(self, skel=None):
    #     """
    #     Return a 1-dimensional vector of the skeleton's state, as defined by
    #     the state code. When skeleton is not specified, control_skel is used
    #     """

    #     if skel is None:
    #         skel = self.robot_skeleton

    #     state = np.array([self.framenum / self.num_frames])

    #     for dof_name in self._dof_names:

    #         indices, body_index = self.metadict[dof_name]
    #         body = skel.bodynodes[body_index]
    #         fi, li = indices[0], indices[-1] + 1

    #         if dof_name != ROOT_KEY:
    #             if len(indices) > 1:

    #                 converted_angle = self.angle_to_rep(pad2length(skel.q[fi:li],
    #                                                                3))
    #             else:
    #                 converted_angle = skel.q[fi:fi+1]
    #         else:
    #             converted_angle = self.angle_to_rep(skel.q[0:3])
    #             fi, li = 0, 4

    #         # TODO Pass in an actual angular velocity instead of dq
    #         state = np.concatenate([state,
    #                                 body.com() - skel.bodynodes[0].com(),
    #                                 converted_angle,
    #                                 body.dC,
    #                                 skel.dq[fi:li]])

    #     return state

    # def quaternion_angles(self, skel):

    #     angles = [None] * len(self._dof_names)

    #     for dof_index, dof_name in enumerate(self._dof_names):

    #         indices, _ = self.metadict[dof_name]

    #         if dof_name != ROOT_KEY:
    #             euler_angle = pad2length(skel.q[indices[0]:indices[-1]+1],
    #                                      3)
    #         else:
    #             euler_angle = skel.q[0:3]

    #         angles[dof_index] = euler2quat(*(euler_angle[::-1]))

    #     return np.array(angles)

    # def reward(self, skel, framenum):

    #     angles = self.quaternion_angles(skel)

    #     ref_angles = self.ref_quat_frames[framenum]
    #     ref_com = self.ref_com_frames[framenum]
    #     ref_ee_positions = self.ref_ee_frames[framenum]

    #     #####################
    #     # POSITIONAL REWARD #
    #     #####################

    #     quatdiffs = [mult(inverse(ra), a) for a, ra in zip(angles,
    #                                                        ref_angles)]
    #     posdiffs = [2 * atan2(norm(quat[1:]), quat[0]) for quat in quatdiffs]

    #     posdiffmag = norm(posdiffs)**2

    #     ###################
    #     # VELOCITY REWARD #
    #     ###################

    #     ref_dq = self.RefDQs[framenum]
    #     veldiffmag = norm(skel.dq - ref_dq)**2

    #     #######################
    #     # END EFFECTOR REWARD #
    #     #######################

    #     eediffmag = norm(self._get_ee_positions(skel)
    #                      - self.ref_ee_frames[framenum])**2

    #     #########################
    #     # CENTER OF MASS REWARD #
    #     #########################

    #     comdiffmag = norm(skel.com() - ref_com)**2

    #     ################
    #     # TOTAL REWARD #
    #     ################

    #     diffmags = [posdiffmag, veldiffmag, eediffmag, comdiffmag]

    #     reward = sum([ow * exp(iw * diff)
    #                   for ow, iw, diff in zip(self._outerweights,
    #                                           self._innerweights,
    #                                           diffmags)])

    #     return reward

    # def angles_from_netvector(self, netvector):

    #     target_q = np.zeros(len(self.robot_skeleton.q) - 6)
    #     q_index = 0
    #     nv_index = 0

    #     for dof_name in self._actuated_dof_names:
    #         indices, _ = self.metadict[dof_name]

    #         if len(indices) == 1:
    #             target_q[q_index] = netvector[nv_index:nv_index+1]
    #             q_index += 1
    #             nv_index += 1

    #         else:
    #             raw_angle = netvector[nv_index:nv_index \
    #                                   + ActionMode.lengths[self.actionmode]]
    #             euler_angle = np.array(self.angle_from_rep(raw_angle))
    #             target_q[q_index:q_index + len(indices)] \
    #                 = euler_angle[:len(indices)]

    #             q_index += len(indices)
    #             nv_index += ActionMode.lengths[self.actionmode]

    #     if q_index != len(self.robot_skeleton.q) - 6:
    #         raise RuntimeError("Not all dofs mapped over")
    #     if nv_index != len(netvector):
    #         raise RuntimeError("Not all net outputs used")

    #     return target_q


    # def should_terminate(self, newstate):
    #     done = self.framenum >= self.num_frames
    #     done = done or reward < self.reward_cutoff
    #     return done


    # def PID(self, skel, dof_targets):
    #     """
    #     Targets should be all for ACTUATED dofs (meaning all of them will be
    #     used)
    #     """

    #     tau = self.p_gain * (dof_targets - skel.q[6:]) \
    #           - self.d_gain * (skel.dq[6:])
    #     tau = np.clip(tau, -self.max_torque, self.max_torque)

    #     return tau

    #################################
    # UNIMPORTANT RENDERING METHODS #
    #################################

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
