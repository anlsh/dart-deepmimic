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

class JointType:

    TRANS = 0
    ROT = 1
    FREE = 2

def pad2length(vector, length):
    padded = np.zeros(length)
    padded[:len(vector)] = deepcopy(vector)
    return padded

def get_metadict(skel, type_lambda):

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
        metadict[dof_name] = (indices, child_body_index,
                              type_lambda(dof_name))

    return metadict


class DartDeepMimicEnv(dart_env.DartEnv):

    def __init__(self,
                 skel_path,
                 mocap_path,
                 # policy_query_frequency,
                 # refmotion_dt,
                 statemode,
                 actionmode,
                 # p_gain, d_gain,
                 pos_noise, vel_noise,
                 # reward_cutoff,
                 pos_weight, pos_decay,
                 vel_weight, vel_decay,
                 ee_weight, ee_decay,
                 com_weight, com_decay,
                 # max_torque,
                 # max_angle,
                 delta_actions,
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
        self.actionmode = actionmode
        self.delta_actions = delta_actions

        self.angle_to_rep = lambda x: None
        self.angle_from_rep = lambda x: None

        if self.statemode == StateMode.GEN_EULER:
            self.angle_to_rep = lambda x: x
        elif self.statemode == StateMode.GEN_QUAT:
            self.angle_to_rep = lambda theta: euler2quat(z=theta[2],
                                                         y=theta[1],
                                                         x=theta[0])
        elif self.statemode == StateMode.GEN_AXIS:
            raise NotImplementedError()

        if self.actionmode == ActionMode.GEN_EULER:
            self.angle_from_rep = lambda x: x

        elif self.actionmode == ActionMode.GEN_QUAT:
            raise NotImplementedError()

        elif self.actionmode == ActionMode.GEN_AXIS:
            self.angle_from_rep = lambda aa: angle_axis2euler(theta=aa[0],
                                                             vector=aa[1:])[::-1]


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
        self.mocap_path = mocap_path
        # TODO Make sure that -1 is the right skel to use: CLI parameter?
        ref_skel = pydart.World(.0001, self.skel_path).skeletons[-1]

        # The lambda should, given a joint name, return a JointType code
        self.metadict = get_metadict(ref_skel, self.type_lambda)

        self._dof_names = [key for key in self.metadict]
        self._dof_names.sort(key=lambda x:self.metadict[x][0][0])
        self._rotational_dof_names = [name for name in self._dof_names
                                      if self.type_lambda(name) == JointType.ROT]
        # TODO Find a better way to distinguish actuated joints?
        self._actuated_dof_names = [name for name in self._dof_names
                                    if not name.startswith(ROOT_KEY)]
        for name in self._actuated_dof_names:
            _, __, joint_type = self.metadict[name]
            if joint_type != JointType.ROT:
                # TODO Support non-rotational actuated joints?
                raise NotImplementedError("Non-rot actuated joints unsupported")

        #####################################
        # Parse reference mocap information #
        #####################################

        self.RefQs, self.RefDQs, self.RefQuats, self.RefEEs, \
            self.RefComs = self.construct_frames(ref_skel,
                                                 self.mocap_path)
        self.num_frames = len(self.RefQs)

        ############################################
        # Calculate observation, action dimensions #
        ############################################

        self.obs_dim = len(self._get_obs(ref_skel))
        self.action_dim = sum([ActionMode.lengths[self.actionmode]
                               if len(self.metadict[name][0]) > 1 else 1
                               for name in self._actuated_dof_names])

        # The control bounds don't actually do anything lol, they're just
        # there to give the environment dimensions (according to Visak)
        control_bounds = np.array([10*np.ones(self.action_dim,),
                                   -10*np.ones(self.action_dim,)])

        dart_env.DartEnv.__init__(self,
                                  [self.skel_path],
                                  self.action_dim,
                                  self.obs_dim,
                                  control_bounds,
                                  disableViewer=False)

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
        # Type of angles in action space

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

    def construct_frames(self, ref_skel, ref_motion_path):

        with open(ref_motion_path, "rb") as fp:
            RefQs = np.loadtxt(fp)

        num_frames = len(RefQs)

        # TODO I need to parse velocities from the motion capture data!!
        # TODO Also, figure out why info I parse doesn't line w/ Visak's
        RefDQs = [-1] * num_frames
        RefQuats = [None] * num_frames
        RefComs = [None] * num_frames
        RefEEs = [None] * num_frames

        for i in range(num_frames):

            ref_skel.set_positions(RefQs[i])

            RefQuats[i] = self.quaternion_angles(ref_skel)
            # TODO TBH I'm still not sure bodynodes[0] is the thing to use
            RefComs[i] = ref_skel.bodynodes[0].com()
            RefEEs[i] = self._get_ee_positions(ref_skel)

        # TODO Should I use np.array on RefQs?
        return (RefQs,
                np.array(RefDQs),
                np.array(RefQuats),
                np.array(RefEEs),
                np.array(RefComs))

    def type_lambda(self, joint_name):
        raise NotImplementedError()

    def target_angles(self, actuated_angles):
        """
        Given a set of actuated angles, return the actuated angle targets
        that we'll try to PID to
        """
        if self.delta_actions:
            return self.RefQs[self.framenum][6:] + actuated_angles
        else:
            return actuated_angles

    def pos_diff(self, skel, framenum):

        quats = self.quaternion_angles(skel)
        refquats = self.RefQuats[framenum]

        quatdiffs = [mult(inverse(ra), a) for a, ra in zip(quats,
                                                           refquats)]
        # TODO Doesnt the Wikipedia page say to use atan2?
        posdiffs = [2*np.arccos(quat[0]) for quat in quatdiffs]

        # TODO Enforce a finiteness check on the results!!
        return np.sum(np.square(posdiffs))

    def vel_diff(self, skel, framenum):

        # TODO I can just use [i] instead of [i,:], right?
        return np.sum(np.square(self.skel.dq - self.RefDQs[framenum]))

    def ee_diff(self, skel, framenum):

        offsets = self._get_ee_positions(skel) - self.RefEEs[framenum]
        return np.sum(np.square(offsets))

    def com_diff(self, skel, framenum):
        # TODO TBH I'm still not sure bodynodes[0] is the thing to use
        return np.sum(np.square(self.RefComs[framenum] - skel.bodynodes[0].com()))

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
        target[6:] = self.target_angles(self.angles_from_netvector(nvec))

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
        if default is not None:
            return default
        else:
            return self.random.randint(0, self.num_frames - 1)

    def reset(self, framenum=None, noise=True):

        pnoise = int(noise) * self.pos_noise
        vnoise = int(noise) * self.vel_noise

        self.dart_world.reset()

        self.framenum = self.get_random_framenum(framenum)

        qpos = self.RefQs[self.framenum,
                          :].reshape(self.robot_skeleton.ndofs) \
               + self.np_random.uniform(low=-pnoise, high=pnoise,
                                        size=self.robot_skeleton.ndofs)

        qvel = self.RefDQs[self.framenum,
                           :].reshape(self.robot_skeleton.ndofs) \
               + self.np_random.uniform(low=-vnoise, high=vnoise,
                                        size=self.robot_skeleton.ndofs)

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_ee_positions(self, skel):
        """
        Return a numpy array w/ each row being position of an ee
        """
        # TODO I'd like to parse ee indices and offsets myself one day
        # bit of a pipe dream IMO but it's certainly ideal
        raise NotImplementedError()

    def _get_obs(self, skel=None):

        if skel is None:
            skel = self.robot_skeleton

        state = np.array([self.framenum / self.num_frames])

        for dof_name in self._dof_names:

            indices, body_index, joint_type = self.metadict[dof_name]
            body = skel.bodynodes[body_index]
            fi, li = indices[0], indices[-1] + 1

            tpos = None
            # TODO Pass in an actual angular velocity instead of dq
            tvel = skel.dq[fi:li]
            # TODO TBH I'm still not sure bodynodes[0] is the thing to use
            bpos = body.com() - skel.bodynodes[0].com()
            bvel = body.dC

            if joint_type == JointType.TRANS:
                tpos = skel.q[fi:li]

            elif joint_type == JointType.ROT:
                if len(indices) > 1:
                    padded_angle = pad2length(skel.q[fi:li], 3)
                    tpos = self.angle_to_rep(padded_angle)
                else:
                    tpos = skel.q[fi:fi+1]

            elif joint_type == JointType.FREE:
                raise NotImplementedError()
            else:
                raise RuntimeError("Unrecognized joint type!")

            state = np.concatenate([state,
                                    bpos, tpos, bvel, tvel])

        return state

    def quaternion_angles(self, skel):

        angles = [None] * len(self._rotational_dof_names)

        for quat_index, dof_name in enumerate(self._rotational_dof_names):

            indices, _, __ = self.metadict[dof_name]

            euler_angle = pad2length(skel.q[indices[0]:indices[-1]+1],
                                     3)

            angles[quat_index] = euler2quat(z=euler_angle[2],
                                            y=euler_angle[1],
                                            x=euler_angle[0])

        return np.array(angles)

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

    def angles_from_netvector(self, netvector):
        """
        Given a neural network output, return a set of target angle for
        the actuated rotational degrees of freedom
        """
        # TODO Eventually should allow targets for translational dofs too?

        target_q = np.zeros(len(self.robot_skeleton.q) - 6)
        q_index = 0
        nv_index = 0

        for dof_name in self._actuated_dof_names:
            indices, _, __ = self.metadict[dof_name]

            if len(indices) == 1:
                target_q[q_index] = netvector[nv_index:nv_index+1]
                q_index += 1
                nv_index += 1

            else:
                raw_angle = netvector[nv_index:nv_index \
                                      + ActionMode.lengths[self.actionmode]]
                euler_angle = np.array(self.angle_from_rep(raw_angle))
                target_q[q_index:q_index + len(indices)] \
                    = euler_angle[:len(indices)]

                q_index += len(indices)
                nv_index += ActionMode.lengths[self.actionmode]

        # TODO This check has never failed on me, so can prolly delete it
        if q_index != len(self.robot_skeleton.q) - 6:
            raise RuntimeError("Not all dofs mapped over")
        if nv_index != len(netvector):
            raise RuntimeError("Not all net outputs used")

        return target_q

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
