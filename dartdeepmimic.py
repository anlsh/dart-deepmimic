import pdb
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
from euclideanSpace import angle_axis2euler, euler2quat, quat2euler
from quaternions import mult, inverse

# Customizable parameters
ROOT_THETA_KEY = "root"

# ROOT_KEY isn't customizeable. It should correspond
# to the name of the root node in the amc (which is usually "root")
ROOT_KEY = "root"
GRAVITY_VECTOR = np.array([0, -9.8, 0])

END_OFFSET = np.array([1, 1, 1])


class StateMode:
    """
    Just a convenience enum
    """
    GEN_EULER = 0
    GEN_QUAT = 1
    GEN_AXIS = 2

    MIX_EULER = 3
    MIX_QUAT = 4
    MIX_AXIS = 5


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

def normalize(vector, identity=None):

    if np.linalg.norm(vector) == 0:
        if identity is None:
            raise RuntimeError("Tried to normalize a 0 vector")
        else:
            return identity
    else:
        return np.divide(vector, np.linalg.norm(vector))


def get_metadict(skel):
    """
    Given a skeleton object, create a dictionary mapping each (actuated)
    joint name to (list of indices it occupies in skel.q, child body)
    """
    joint_names = [joint.name for joint in skel.joints]
    skel_dofs = skel.dofs

    metadict = {}
    for dof_name in joint_names:
        indices = [i for i, dof in enumerate(skel_dofs)
                   if dof.name.startswith(dof_name)]
        if len(indices) == 0:
            # Some joints (like welds) dont have dofs and so won't appear
            # we avoid adding those to the dict entirely
            continue
        child_body = [body for body in skel.bodynodes
                      if body.name.startswith(dof_name)][0]
        metadict[dof_name] = (indices, child_body)

    return metadict


def map_dofs(dof_list, pos_list, vel_list, pstdv, vstdv):
    """
    Given a list of dof objects, set their positions and velocities
    accordingly
    """
    if not(len(dof_list) == len(pos_list) == len(vel_list)):
        raise RuntimeError("Zip got tattered lists")
    for dof, pos, vel in zip(dof_list, pos_list, vel_list):
        pos = np.random.normal(pos, pstdv)
        vel = np.random.normal(vel, vstdv)

        dof.set_position(float(pos))
        dof.set_velocity(float(vel))


def sd2rr(rvector):
    """
    Takes a vector of sequential degrees and returns the rotation it
    describes in rotating radians (the format DART expects angles in)
    """

    rvector = np.multiply(rvector, pi / 180)

    rmatrix = compose_matrix(angles=rvector, angle_order="sxyz")
    return euler_from_matrix(rmatrix[:3, :3], axes="rxyz")


def euler_velocity(final, initial, dt):
    """
    Given two xyz euler angles (sequentian degrees)
    Return the euler angle velocity (in rotating radians i think)
    """
    # TODO IT'S NOT RIGHT AAAAHHHH
    return np.divide(sd2rr(np.subtract(final, initial)), dt)


class DartDeepMimicEnv(dart_env.DartEnv):

    def __init__(self, control_skeleton_path,
                 reference_motion_path, refmotion_dt,
                 statemode,
                 actionmode,
                 p_gain, d_gain,
                 pos_init_noise, vel_init_noise,
                 reward_cutoff,
                 pos_weight, pos_inner_weight,
                 vel_weight, vel_inner_weight,
                 ee_weight, ee_inner_weight,
                 com_weight, com_inner_weight,
                 max_torque,
                 max_angle, default_damping,
                 default_spring,
                 visualize, simsteps_per_dataframe,
                 screen_width,
                 screen_height,
                 gravity, self_collide):

        #######################################
        # Just set a bunch of self.parameters #
        #######################################

        self.statemode = statemode
        self.actionmode = actionmode
        self.refmotion_dt = refmotion_dt
        self.simsteps_per_dataframe = simsteps_per_dataframe
        self.pos_init_noise = pos_init_noise
        self.vel_init_noise = vel_init_noise
        self.max_torque = max_torque
        self.max_angle = max_angle
        self.default_damping = default_damping
        self.default_spring = default_spring
        self.reward_cutoff = reward_cutoff
        self.gravity = gravity
        if not self.gravity:
            warnings.warn("Gravity is disabled, be sure you meant to do this!",
                          RuntimeWarning)
        self.self_collide = self_collide
        if not self.self_collide:
            warnings.warn("Self collisions are disabled, be sure you meant"
                          + " to do this!", RuntimeWarning)
        self.p_gain = p_gain
        self.d_gain = d_gain
        if (self.p_gain < 0) or (self.d_gain < 0):
            raise RuntimeError("All PID gains should be positive")

        if (pos_inner_weight > 0) or (vel_inner_weight > 0) or \
           (ee_inner_weight > 0) or (com_inner_weight) > 0:
            raise RuntimeError("Inner weights should always be <= 0")
        if (pos_weight < 0) or (vel_weight < 0) or \
           (ee_weight < 0) or (com_weight) < 0:
            raise RuntimeError("Outer weights should always be >= 0")

        self.__visualize = visualize

        self.pos_weight = pos_weight
        self.pos_inner_weight = pos_inner_weight
        self.vel_weight = vel_weight
        self.vel_inner_weight = vel_inner_weight
        self.ee_weight = ee_weight
        self.ee_inner_weight = ee_inner_weight
        self.com_weight = com_weight
        self.com_inner_weight = com_inner_weight

        self._outerweights = [self.pos_weight, self.vel_weight,
                              self.ee_weight, self.com_weight]

        self._innerweights = [self.pos_inner_weight,
                              self.vel_inner_weight,
                              self.ee_inner_weight,
                              self.com_inner_weight]

        self._control_skeleton_path = control_skeleton_path
        self.ref_skel = pydart.World(.00001,
                                     control_skeleton_path).skeletons[-1]

        self.metadict = get_metadict(self.ref_skel)

        # The sorting is critical
        self._dof_names = [key for key in self.metadict]
        self._dof_names.sort(key=lambda x:self.metadict[x][0][0])
        # The first degree of freedom is always the root
        self._actuated_dof_names = self._dof_names[1:]

        self._end_effector_indices = [i for i, node
                                       in enumerate(self.ref_skel.bodynodes)
                                     if len(node.child_bodynodes) == 0]

        self.framenum = 0
        self.num_frames, frames = self.construct_frames(reference_motion_path)
        self.ref_q_frames, self.ref_dq_frames, \
            self.ref_quat_frames, self.ref_com_frames, self.ref_ee_frames \
            = frames

        # Setting of control_skel to ref_skel is just temporary so that
        # load_world can call self._get_obs() and set it correctly afterwards
        self.control_skel = self.ref_skel

        self.obs_dim = len(self._get_obs())
        self.action_dim = 0
        for name in self._actuated_dof_names:
            indices, _ = self.metadict[name]
            self.action_dim += 1 if len(indices) == 1 \
                          else ActionMode.lengths[self.actionmode]

        self.load_world()

    def load_world(self):

        action_limits = self.max_angle * pi * np.ones(self.action_dim)
        action_limits = [-action_limits, action_limits]

        super(DartDeepMimicEnv,
              self).__init__(model_paths=[self._control_skeleton_path],
                             frame_skip=1,
                             observation_size=self.obs_dim,
                             action_bounds=action_limits,
                             dt=self.refmotion_dt / self.simsteps_per_dataframe,
                             visualize=self.__visualize,
                             disableViewer=not self.__visualize)

        self.dart_world.set_gravity(int(self.gravity) * GRAVITY_VECTOR)
        self.control_skel = self.dart_world.skeletons[-1]

        self.control_skel.set_self_collision_check(self.self_collide)

        for joint in self.control_skel.joints:
            if joint.name == ROOT_KEY:
                continue
            for index in range(joint.num_dofs()):
                joint.set_damping_coefficient(index, self.default_damping)
                joint.set_spring_stiffness(index, self.default_spring)


    def construct_frames(self, ref_motion_path):

        raise NotImplementedError()

    def sync_skel_to_frame(self, skel, frame_index, pos_stdv, vel_stdv):
        """
        Given a skeleton and mocap frame index, use self.metadict to sync all
        the dofs. Will work on different skeleton objects as long as they're
        created from the same file (like self.control_skel and self.ref_skel)

        If noise is true, then positions and velocities will have random normal
        noise added to them
        """
        # Set the root position
        # The root pos is never fuzzed (prevent clipping into the
        # ground)
        q = self.ref_q_frames[frame_index]
        dq = self.ref_dq_frames[frame_index]

        map_dofs(skel.dofs[3:6], q[3:6], dq[3:6], 0, 0)
        map_dofs(skel.dofs[0:3], q[:3], dq[:3], 0, vel_stdv)
        map_dofs(skel.dofs[6:], q[6:], dq[6:], pos_stdv, vel_stdv)


    def _get_obs(self, skel=None):
        """
        Return a 1-dimensional vector of the skeleton's state, as defined by
        the state code. When skeleton is not specified, the
        control_skel is used
        """

        if skel is None:
            skel = self.control_skel
        else:
            warnings.warn("_get_obs used w/ non-control skeleton, you sure"
                          + "you know what you're doing?", RuntimeWarning)

        if self.statemode == StateMode.GEN_EULER:
            angle_tform = lambda x: x
        elif self.statemode == StateMode.GEN_QUAT:
            angle_tform = lambda x: euler2quat(*(x[::-1]))
        elif self.statemode == StateMode.GEN_AXIS:
            angle_tform = lambda x: axisangle_from_euler(*x, axes="rxyz")
        else:
            raise RuntimeError("Unimplemented state code: "
                               + str(self.statemode))

        state = np.array([self.framenum / self.num_frames])
        for dof_name in self._dof_names:
            indices, body = self.metadict[dof_name]

            if dof_name != ROOT_KEY:
                if len(indices) > 1:
                    euler_angle = pad2length(skel.q[indices[0]:indices[-1]+1],
                                            3)
                    converted_angle = angle_tform(euler_angle)
                else:
                    converted_angle = skel.q[indices[0]:indices[0]+1]
            else:
                converted_angle = angle_tform(skel.q[0:3])

            relpos = body.com() - skel.com()
            linvel = body.dC
            # TODO Need to convert dq into an angular velocity
            dq = skel.dq[indices[0]:indices[-1]+1]
            state = np.concatenate([state, relpos, converted_angle, linvel, dq])

        return state

    def quaternion_angles(self, skel=None):
        if skel is None:
            skel = self.control_skel

        angles = []

        for dof_name in self._dof_names:

            indices, _ = self.metadict[dof_name]

            if dof_name != ROOT_KEY:
                euler_angle = pad2length(skel.q[indices[0]:indices[-1]+1],
                                         3)
            else:
                euler_angle = skel.q[0:3]

            converted_angle = euler2quat(*(euler_angle[::-1]))
            angles.append(converted_angle)

        return np.array(angles)


    def reward(self, skel, framenum):

        angles = self.quaternion_angles(skel)

        ref_angles = self.ref_quat_frames[framenum]
        ref_com = self.ref_com_frames[framenum]
        ref_ee_positions = self.ref_ee_frames[framenum]

        #####################
        # POSITIONAL REWARD #
        #####################

        posdiff = [2 * np.arccos(mult(inverse(ra), a)[0])
                   for a, ra in zip(angles, ref_angles)]
        posdiffmag = np.sum(np.square(posdiff))

        ###################
        # VELOCITY REWARD #
        ###################

        ref_dq = self.ref_dq_frames[framenum]
        veldiffmag = np.sum(np.square(skel.dq - ref_dq))

        #######################
        # END EFFECTOR REWARD #
        #######################

        eediffmag = np.sum(np.square([norm(self.control_skel.bodynodes[j].to_world(END_OFFSET)
                          - ref_ee_positions[i])
                     for i, j in enumerate(self._end_effector_indices)]))

        #########################
        # CENTER OF MASS REWARD #
        #########################

        comdiffmag = np.sum(np.square(self.control_skel.com() - ref_com))

        ################
        # TOTAL REWARD #
        ################

        diffmags = [posdiffmag, veldiffmag, eediffmag, comdiffmag]

        reward = sum([ow * exp(iw * diff)
                      for ow, iw, diff in zip(self._outerweights,
                                              self._innerweights,
                                              diffmags)])

        return reward

    def q_from_netvector(self, netvector):

        if self.actionmode == ActionMode.GEN_EULER:
            angle_tform = lambda x: x
        elif self.actionmode == ActionMode.GEN_QUAT:
            raise NotImplementedError()
        elif self.actionmode == ActionMode.GEN_AXIS:
            def tform_from_angleaxis(angleaxis):
                angle = angleaxis[0]
                axis = angleaxis[1:]
                if norm(axis) != 0:
                    return angle_axis2euler(angle, axis)[::-1]
                else:
                    return np.array([0, 0, 0])
            angle_tform = tform_from_angleaxis
        else:
            raise RuntimeError("Unimplemented action code: "
                               + str(self.actionmode))

        q = np.zeros(len(self.control_skel.q))
        q_index = 6
        netvector_index = 0
        for dof_name in self._actuated_dof_names:
            indices, _ = self.metadict[dof_name]

            if len(indices) == 1:
                q[q_index] = netvector[netvector_index:netvector_index+1]
                q_index += 1
                netvector_index += 1

            else:
                raw_angle = netvector[netvector_index:netvector_index \
                                      + ActionMode.lengths[self.actionmode]]
                euler_angle = angle_tform(raw_angle)
                q[q_index:q_index + len(indices)] = euler_angle[:len(indices)]
                q_index += len(indices)
                netvector_index += ActionMode.lengths[self.actionmode]

        if q_index != len(self.ref_skel.q):
            raise RuntimeError("Not all dofs mapped over")
        if netvector_index != len(netvector):
            raise RuntimeError("Not all net outputs used")

        return q


    def should_terminate(self, reward, newstate):
        done = self.framenum == self.num_frames - 1
        done = done or reward < self.reward_cutoff
        return done


    def step(self, action_vector):
        """
        action_vector is of length (anglemodelength) * (num_actuated_joints)
        """
        dof_targets = self.q_from_netvector(action_vector)

        tau = self.p_gain * (dof_targets[6:] - self.control_skel.q[6:]) \
              - self.d_gain * (self.control_skel.dq[6:])
        tau = np.clip(tau, -self.max_torque, self.max_torque)

        self.do_simulation(np.concatenate([np.zeros(6), tau]),
                           self.simsteps_per_dataframe)

        newstate = self._get_obs()
        if not np.isfinite(newstate).all():
            raise RuntimeError("Ran into an infinite state, terminating")
        reward = self.reward(self.control_skel, self.framenum)
        extrainfo = {"dof_targets": dof_targets}
        done = self.should_terminate(reward, newstate)
        self.framenum += 1

        return newstate, reward, done, extrainfo

    def reset(self, framenum=None, pos_stdv=None, vel_stdv=None):

        if pos_stdv is None:
            pos_stdv = self.pos_init_noise
        if vel_stdv is None:
            vel_stdv = self.vel_init_noise

        self.dart_world.reset()

        if framenum is None:
            framenum = random.randint(0, self.num_frames - 1)
        self.framenum = framenum

        self.sync_skel_to_frame(self.control_skel, self.framenum,
                                pos_stdv, vel_stdv)

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -80
        self._get_viewer().scene.tb.trans[1] = -40
        self._get_viewer().scene.tb.trans[0] = 0

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

