import pdb
from amc import AMC
from asf_skeleton import ASF_Skeleton
from gym.envs.dart import dart_env
from joint import expand_angle, compress_angle
from math import exp, pi, atan2
from numpy.linalg import norm
from transformations import compose_matrix, euler_from_matrix
from transformations import quaternion_from_euler, euler_from_quaternion
from transformations import quaternion_multiply, quaternion_conjugate, \
    quaternion_inverse
import argparse
import numpy as np
import pydart2 as pydart
import random
import warnings

# Customizable parameters
ROOT_THETA_KEY = "root"
REFMOTION_DT = 1 / 120

# ROOT_KEY isn't customizeable. It should correspond
# to the name of the root node in the amc (which is usually "root")
ROOT_KEY = "root"
ROOT_THETA_ORDER = "xyz"
GRAVITY_VECTOR = np.array([0, -9.8, 0])


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


def quaternion_difference(a, b):
    return quaternion_multiply(b, quaternion_inverse(a))


def quaternion_rotation_angle(a):
    # Returns the rotation of a quaternion about its axis in radians
    return atan2(norm(a[1:]), a[0])


def get_metadict(amc_frame, skel_dofs, asf):
    """
    @type amc_frame: A list of (joint-name, joint-data) tuples
    @type skel_dofs: The array of skeleton dofs, as given by skel.dofs
    @type asf: An instance of the ASFSkeleton class

    @return: A dictionary which maps dof names APPEARING IN MOCAP DATA to
    tuples where: - the first element is the list of indices the dof occupies
    in skel_dofs - the second element is the joint's angle order (a string such
    as "xz" or "zyx")
    """
    # README EMERGENCY!!!
    # If the output of this function is ever changed so that the number of
    # actuated dofs is no longer given by (size of output dict - 1), then
    # the setting of action_dim in DartDeepMimic will need to be updated

    dof_data = {}
    for dof_name, _ in amc_frame:
        joint = asf.name2joint[dof_name]
        axes_str = joint.dofs
        indices = [i for i, dof in enumerate(skel_dofs)
                   if dof.name.startswith(dof_name)]
        dof_data[dof_name] = (indices, axes_str)

    return dof_data


class DartDeepMimicEnv(dart_env.DartEnv):

    def __init__(self, control_skeleton_path, asf_path,
                 reference_motion_path,
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
                 gravity):

        self.statemode = statemode
        self.actionmode = actionmode
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
            warnings.warn("Gravity is disabled, be sure you meant to do this!", RuntimeWarning)

        self.framenum = 0

        # This parameter doesn't actually does anything
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

        ###########################################################
        # Extract dof info so that states can be converted easily #
        ###########################################################

        self.ref_skel = pydart.World(REFMOTION_DT / self.simsteps_per_dataframe,
                                     control_skeleton_path).skeletons[-1]

        asf = ASF_Skeleton(asf_path)
        raw_framelist = AMC(reference_motion_path).frames
        self.metadict = get_metadict(raw_framelist[0],
                                     self.ref_skel.dofs, asf)

        # The sorting is absolutely critical
        self._dof_names = [key for key in self.metadict]
        self._dof_names.sort(key=lambda x:self.metadict[x][0][0])
        self._actuated_dof_names = self._dof_names[1:]

        self.ref_euler_posframes = None
        self.ref_euler_velframes = None
        self.ref_quat_posframes = None
        self.num_frames = 0

        self.num_frames, frames = self.construct_frames(raw_framelist)
        self.ref_euler_posframes, self.ref_euler_velframes, self.ref_quat_posframes = frames

        self._end_effector_indices = [i for i, node
                                       in enumerate(self.ref_skel.bodynodes)
                                     if len(node.child_bodynodes) == 0]

        # TODO End reliance on underlying asf
        self._end_effector_offsets = [asf.name2joint[
            self.ref_skel.bodynodes[i].name[:-5]].offset
                                       for i in self._end_effector_indices]

        ################################################
        # Do some calculations related to action space #
        ################################################

        # Calculate the size of the neural network output vector
        self.action_dim = ActionMode.lengths[actionmode] \
                          * (len(self._actuated_dof_names))

        self.control_skel = self.ref_skel
        self.load_world()

        ##################################
        # Simulation stuff for DeepMimic #
        ##################################

        self.p_gain = p_gain
        self.d_gain = d_gain

        if self.p_gain < 0 or self.d_gain < 0:
            raise RuntimeError("All PID gains should be positive")

    def load_world(self):

        action_limits = self.max_angle * pi * np.ones(self.action_dim)
        action_limits = [-action_limits, action_limits]

        super(DartDeepMimicEnv, self).__init__([self._control_skeleton_path],
                                               1,
                                               len(self._get_obs()),
                                               action_limits,
                                               REFMOTION_DT / self.simsteps_per_dataframe,
                                               "parameter",
                                               "continuous",
                                               self.__visualize,
                                               not self.__visualize)
        self.dart_world.set_gravity(int(self.gravity) * GRAVITY_VECTOR)
        self.control_skel = self.dart_world.skeletons[-1]

        # TODO Enable self collisions
        # Have to explicitly enable self collisions
        # self.control_skel.set_self_collision_check(True)

        for joint in self.control_skel.joints:
            if joint.name == ROOT_KEY:
                continue
            for index in range(joint.num_dofs()):
                joint.set_damping_coefficient(index, self.default_damping)
                joint.set_spring_stiffness(index, self.default_spring)

    def construct_frames(self, raw_framelist):
        """
        AMC data is given in sequential degrees, while dart specifies angles
        in rotating radians. The conversion is quite expensive, so we precomute
        all positions and velocities and store the results
        """

        def sd2rr(rvector):
            """
            Takes a vector of sequential degrees and returns the rotation it
            describes in rotating radians (the format DART expects angles in)
            """

            rvector = np.multiply(rvector, pi / 180)

            rmatrix = compose_matrix(angles=rvector, angle_order="sxyz")
            return euler_from_matrix(rmatrix[:3, :3], axes="rxyz")

        num_frames = len(raw_framelist)
        elements_per_frame = len(raw_framelist[0])

        pos_frames = [None] * num_frames
        vel_framelist = [None] * num_frames
        quat_frames = [None] * num_frames

        for i in range(len(raw_framelist)):
            old_i = i - 1 if i > 0 else 0

            current_frame = raw_framelist[i]
            old_frame = raw_framelist[old_i]

            pos_frame = [None] * elements_per_frame
            vel_frame = [None] * elements_per_frame

            # Root data is a little bit special, so we handle it here
            curr_root_data = np.array(current_frame[0][1])
            curr_root_pos, curr_root_theta = curr_root_data[:3], curr_root_data[3:]
            old_root_data = np.array(old_frame[0][1])
            old_root_pos, old_root_theta = old_root_data[:3], old_root_data[3:]
            pos_frame[0] = (ROOT_KEY,
                            np.concatenate([curr_root_pos,
                                            sd2rr(curr_root_theta)]))
            # TODO EMERGENCY Validate that this subtraction of angles works...
            vel_frame[0] = (ROOT_KEY,
                            np.concatenate([np.subtract(curr_root_pos,
                                                        old_root_pos),
                                            sd2rr(np.subtract(curr_root_theta,
                                                              old_root_theta))])
                            / REFMOTION_DT)

            # Deal with the non-root joints in full generality
            joint_index = 0
            for joint_name, curr_joint_angles in current_frame[1:]:
                joint_index += 1
                _, order = self.metadict[joint_name]

                curr_theta = expand_angle(curr_joint_angles, order)
                old_theta = expand_angle(old_frame[joint_index][1], order)

                current_rotation_euler = compress_angle(sd2rr(curr_theta), order)
                velocity_rotation_euler = compress_angle(sd2rr(np.subtract(curr_theta,
                                                                           old_theta))) / REFMOTION_DT

                pos_frame[joint_index] = (joint_name, current_rotation_euler)
                vel_frame[joint_index] = (joint_name, velocity_rotation_euler)

            pos_frames[i] = pos_frame
            vel_framelist[i] = vel_frame

            self.sync_skel_to_frame(self.ref_skel, None, 0, 0,
                                    pos_frame, vel_frame)
            quat_frames[i] = self.gencoordtuple_as_pos_and_qautlist(self.ref_skel)

        return num_frames, (pos_frames, vel_framelist, quat_frames)

    def sync_skel_to_frame(self, skel, frame_index, pos_stdv, vel_stdv,
                           pos_frame=None, vel_frame=None):
        """
        Given a skeleton and mocap frame index, use self.metadict to sync all
        the dofs. Will work on different skeleton objects as long as they're
        created from the same file (like self.control_skel and self.ref_skel)

        If noise is true, then positions and velocities will have random normal
        noise added to them
        """
        if frame_index is not None:
            pos_frame = self.ref_euler_posframes[frame_index]
            vel_frame = self.ref_euler_velframes[frame_index]

        def map_dofs(dof_list, pos_list, vel_list, pstdv, vstdv):
            """
            Given a list of dof objects, set their positions and velocities
            accordingly
            """
            for dof, pos, vel in zip(dof_list, pos_list, vel_list):
                pos = np.random.normal(pos, pstdv)
                vel = np.random.normal(vel, vstdv)

                dof.set_position(float(pos))
                dof.set_velocity(float(vel))

        # World to root joint is a bit special so we handle it here...
        root_pos_data = pos_frame[0][1]
        root_vel_data = vel_frame[0][1]
        map_dofs(skel.dofs[3:6], root_pos_data[:3], root_vel_data[:3],
                 pos_stdv, vel_stdv)
        # The root pos is never fuzzed (mostly to prevent clipping into the
        # ground)
        map_dofs(skel.dofs[0:3], root_pos_data[3:], root_vel_data[3:],
                 0, vel_stdv)

        joint_index = 0
        # And handle the rest of the dofs normally
        for joint_name, joint_angles in pos_frame[1:]:
            joint_index += 1
            dof_indices, order = self.metadict[joint_name]
            start_index, end_index = dof_indices[0], dof_indices[-1]

            joint_velocities = vel_frame[joint_index][1]

            map_dofs(skel.dofs[start_index : end_index + 1],
                     joint_angles, joint_velocities, pos_stdv, vel_stdv)


    def _get_obs(self, skel=None):
        """
        Return a 1-dimensional vector of the skeleton's state, as defined by
        the state code. When skeleton is not specified, the
        control_skel is used
        """

        if skel is None:
            skel = self.control_skel
        else:
            # There's not any reason I can think of to actually use any other
            # skeleton, the print method is just here so I never accidentally
            # replace it
            warnings.warn("_get_obs used w/ non-control skeleton, you sure you know what you're doing?", RuntimeWarning)

        if self.statemode == StateMode.GEN_EULER:
            state = self.gencoordtuple_as_pos_and_eulerlist(
                skel)
        elif self.statemode == StateMode.GEN_QUAT:
            state = self.gencoordtuple_as_pos_and_qautlist(
                skel)
        elif self.statemode == StateMode.GEN_AXIS:
            state = self.gencoordtuple_as_pos_and_axisanglelist(
                skel)
        else:
            raise RuntimeError("Unimplemented state code: "
                               + str(self.statemode))

        pos, vel = state
        posvector = np.concatenate([pos[0], np.concatenate(pos[1])])
        velvector = np.concatenate([vel[0], np.concatenate(vel[1])])
        return np.concatenate([posvector, velvector])


    def gencoordtuple_as_pos_and_eulerlist(self, skeleton):
        """
        @type skeleton: A dart skeleton

        @return: A tuple where first component is positional info, second is
        velocity info


        Each component is itself a tuple where the first element is an array
        containing the translational component, and the second is a list of
        fully-specified euler angles for each of the dofs (in a consistent but
        opaque order)
        """
        def _genq_to_pos_and_eulerdict(generalized_q):
            """
            @type generalized_q: A vector of dof values, as given by skel.q or .dq

            @return: A tuple where
                - index 0 is the root positional component
                - [1] is a dictionary mapping dof names to fully specified euler
                angles in xyz order
            @type: Tuple
            """

            root_translation = generalized_q[3:6]
            expanded_angles = {ROOT_THETA_KEY: expand_angle(generalized_q[0:3],
                                                            ROOT_THETA_ORDER)}
            for dof_name in self._actuated_dof_names:
                indices, order = self.metadict[dof_name]
                fi = indices[0]
                li = indices[-1]
                expanded_angles[dof_name] = expand_angle(generalized_q[fi:li + 1],
                                                        order)
            return root_translation, expanded_angles

        pos, angles_dict = _genq_to_pos_and_eulerdict(skeleton.q)
        dpos, dangles_dict = _genq_to_pos_and_eulerdict(skeleton.dq)

        angles = np.array([angles_dict[key] for key in self._dof_names])
        dangles = np.array([dangles_dict[key] for key in self._dof_names])

        return (pos, angles), (dpos, dangles)


    def gencoordtuple_as_pos_and_qautlist(self, skel):
        """
        Same as gencoordtuple_as_pos_and_eulerlist, but with the angles
        converted to quaternions
        """

        pos_info, vel_info = self.gencoordtuple_as_pos_and_eulerlist(skel)
        pos, angles = pos_info
        dpos, dangles = vel_info

        angles = [quaternion_from_euler(*t, axes="rxyz") for t in angles]
        dangles = [quaternion_from_euler(*t, axes="rxyz") for t in dangles]

        return (pos, angles), (dpos, dangles)

    def gencoordtuple_as_pos_and_axisanglelist(self, skel):
        """
        Same as gencoordtuple_as_pos_and_eulerlist, but with the angles
        converted to axisangle
        """
        raise NotImplementedError()

    def reward(self, skel, framenum):

        self.sync_skel_to_frame(self.ref_skel, framenum, 0, 0)
        pos, vel = self.gencoordtuple_as_pos_and_qautlist(skel)
        refpos, refvel = self.ref_quat_posframes[framenum]

        pos, angles = pos
        dpos, dangles = vel

        refpos, refangles = refpos
        drefpos, drefangles = refvel

        #####################
        # POSITIONAL REWARD #
        #####################

        posdiff = [quaternion_difference(ra, a)
                   for a, ra in zip(angles, refangles)]
        posdiffmag = sum([quaternion_rotation_angle(d)**2 for d in posdiff])

        ###################
        # VELOCITY REWARD #
        ###################

        velocity_quats = zip(dangles, drefangles)
        velocity_error = [new - old for new, old in velocity_quats]

        veldiffmag = sum([norm(v)**2 for v in velocity_error])

        #######################
        # END EFFECTOR REWARD #
        #######################

        # TODO THe units are off, the paper specifically specifies units of
        # meters
        eediffmag = sum([norm(self.control_skel.bodynodes[i].to_world(offset)
                              - self.ref_skel.bodynodes[i].to_world(offset))**2
                         for i, offset in zip(self._end_effector_indices,
                                              self._end_effector_offsets)])

        #########################
        # CENTER OF MASS REWARD #
        #########################

        comdiffmag = norm(self.control_skel.com() - self.ref_skel.com())**2

        ################
        # TOTAL REWARD #
        ################

        diffmags = [posdiffmag, veldiffmag, eediffmag, comdiffmag]

        reward = sum([ow * exp(iw * diff)
                      for ow, iw, diff in zip(self._outerweights,
                                              self._innerweights,
                                              diffmags)])

        return reward

    def _expanded_euler_from_action(self, raw_action):
        """
        Given a 1-dimensional vector representing a neural network output,
        construct from it a set of target angles for the ACTUATED degrees of
        freedom
        (the ones in metadict, minus the root)

        Because of how action_dim is defined up in __init__, raw_action
        should always have the correct dimensions
        """

        # Reshape into a list of 3-vectors
        output_angles = np.reshape(raw_action,
                                   (-1, ActionMode.lengths[self.actionmode]))

        # TODO Normalize and validate values to make sure they're actually
        # valid euler angles / quaternions / whatever
        if self.actionmode == ActionMode.GEN_EULER:
            return output_angles
        elif self.actionmode == ActionMode.GEN_QUAT:
            return [euler_from_quaternion(*t, axes="rxyz")
                    for t in output_angles]
        elif self.actionmode == ActionMode.GEN_AXIS:
            return [euler_from_axisangle(*t, axes="rxyz")
                    for t in output_angles]
        else:
            raise RuntimeError("Unrecognized or unimplemented action code: "
                               + str(self.actionmode))

    def _expanded_euler_to_dofvector(self, expanded_euler):

        if len(expanded_euler) != len(self._actuated_dof_names):
            raise RuntimeError("Mismatch between number of actuated dofs and angles passed in")
        # expanded_euler calculated correctly
        # print("Calculated Angle Targets\n", expanded_euler)

        ret = np.concatenate([compress_angle(expanded_euler[i],
                                             self.metadict[key][1])
                              for i, key in enumerate(self._actuated_dof_names)])
        return ret

    def step(self, action_vector):
        """
        action_vector is of length (anglemodelength) * (num_actuated_joints)
        """

        expanded_target_euler = self._expanded_euler_from_action(action_vector)
        dof_targets = self._expanded_euler_to_dofvector(expanded_target_euler)

        tau = self.p_gain * (dof_targets - self.control_skel.q[6:]) \
              - self.d_gain * (self.control_skel.dq[6:])
        tau = np.clip(tau, -self.max_torque, self.max_torque)

        self.do_simulation(np.concatenate([np.zeros(6), tau]),
                           self.simsteps_per_dataframe)

        newstate = self._get_obs()
        reward = self.reward(self.control_skel, self.framenum)
        extrainfo = {"dof_targets": dof_targets}
        done = self.framenum == self.num_frames - 1 \
               or (reward < self.reward_cutoff)
        if not np.isfinite(newstate).all():
            raise RuntimeError("Ran into an infinite state, terminating")

        self.framenum += 1

        return newstate, reward, done, extrainfo

    def reset(self, framenum=None, pos_stdv=None, vel_stdv=None):

        if pos_stdv is None:
            pos_stdv = self.pos_init_noise
        if vel_stdv is None:
            vel_stdv = self.vel_init_noise

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

class DartDeepMimicArgParse(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()
        self.add_argument('--control-skel-path', required=True,
                          help='Path to the control skeleton')
        self.add_argument('--asf-path', required=True,
                          help='Path to asf which the skeleton was parsed from')
        self.add_argument('--ref-motion-path', required=True,
                          help='Path to the reference motion AMC')
        self.add_argument('--state-mode', default=0, type=int,
                          help="Code for the state representation")
        self.add_argument('--action-mode', default=0, type=int,
                          help="Code for the action representation")
        self.add_argument('--visualize', default=False,
                          help="DOESN'T DO ANYTHING RIGHT NOW: True if you want a window to render to")
        self.add_argument('--max-torque', type=float, default=90,
                          help="Maximum torque")
        self.add_argument('--max-angle', type=float, default=1000,
                          help="Max magnitude of angle (in terms of pi) that PID can output")
        self.add_argument('--default-damping', type=float, default=80,
                          help="Default damping coefficient for joints")
        self.add_argument('--default-spring', type=float, default=0,
                          help="Default spring stiffness for joints")
        self.add_argument('--simsteps-per-dataframe', type=int, default=10,
                          help="Number of simulation steps per frame of mocap" +
                          " data")
        self.add_argument('--reward-cutoff', type=float, default=0.1,
                          help="Terminate the episode when rewards below this " +
                          "threshold are calculated. Should be in range (0, 1)")
        self.add_argument('--window-width', type=int, default=80,
                          help="Window width")
        self.add_argument('--window-height', type=int, default=45,
                          help="Window height")

        self.add_argument('--pos-init-noise', type=float, default=.05,
                          help="Standard deviation of the position init noise")
        self.add_argument('--vel-init-noise', type=float, default=.05,
                          help="Standart deviation of the velocity init noise")

        self.add_argument('--pos-weight', type=float, default=.65,
                          help="Weighting for the pos difference in the reward")
        self.add_argument('--pos-inner-weight', type=float, default=-2,
                          help="Coefficient for pos difference exponentiation in reward")

        self.add_argument('--vel-weight', type=float, default=.1,
                          help="Weighting for the pos difference in the reward")
        self.add_argument('--vel-inner-weight', type=float, default=-.1,
                          help="Coefficient for vel difference exponentiation in reward")

        self.add_argument('--ee-weight', type=float, default=.15,
                          help="Weighting for the pos difference in the reward")
        self.add_argument('--ee-inner-weight', type=float, default=-40,
                          help="Coefficient for pos difference exponentiation in reward")

        self.add_argument('--com-weight', type=float, default=.1,
                          help="Weighting for the com difference in the reward")
        self.add_argument('--com-inner-weight', type=float, default=-10,
                          help="Coefficient for com difference exponentiation in reward")

        self.add_argument('--p-gain', type=float, default=300,
                            help="P for the PD controller")
        self.add_argument('--d-gain', type=float, default=50,
                          help="D for the PD controller")

        gravity_group = self.add_mutually_exclusive_group()
        gravity_group.add_argument('--gravity', dest='gravity', action='store_true')
        gravity_group.add_argument('--no-gravity', dest='gravity', action='store_false')
        self.set_defaults(gravity=True, help="Whether to enable gravity in the world")

        self.args = None

    def parse_args(self):
        self.args = super().parse_args()
        return self.args

    def get_env(self):

        return DartDeepMimicEnv(control_skeleton_path=self.args.control_skel_path,
                                asf_path=self.args.asf_path,
                                reference_motion_path=self.args.ref_motion_path,
                                statemode=self.args.state_mode,
                                actionmode=self.args.action_mode,
                                p_gain=self.args.p_gain,
                                d_gain=self.args.d_gain,
                                pos_init_noise=self.args.pos_init_noise,
                                vel_init_noise=self.args.vel_init_noise,
                                reward_cutoff=self.args.reward_cutoff,
                                pos_weight=self.args.pos_weight,
                                pos_inner_weight=self.args.pos_inner_weight,
                                vel_weight=self.args.vel_weight,
                                vel_inner_weight=self.args.vel_inner_weight,
                                ee_weight=self.args.ee_weight,
                                ee_inner_weight=self.args.ee_inner_weight,
                                com_weight=self.args.com_weight,
                                com_inner_weight=self.args.com_inner_weight,
                                max_torque=self.args.max_torque,
                                max_angle=self.args.max_angle,
                                default_damping=self.args.default_damping,
                                default_spring=self.args.default_spring,
                                visualize=self.args.visualize,
                                simsteps_per_dataframe=self.args.simsteps_per_dataframe,
                                screen_width=self.args.window_width,
                                screen_height=self.args.window_height,
                                gravity=args.gravity)


if __name__ == "__main__":

    # Don't run this as main, there's really not too much point

    parser = DartDeepMimicArgParse()

    args = parser.parse_args()
    print("Gravity being", args.gravity)

    env = parser.get_env()

    # obs = env.reset(0, False)
    # done = False
    # i = 0
    # while True:
    #     env.render()
    #     obs = env.reset(i, False)
    #     # a = env.action_space.sample()
    #     # state, reward, done, info = env.step(a)
    #     i += 1
    #     if done:
    #         env.reset()

    # for i in range(env.num_frames):
    #     env.reset(i, False)
    #     env.render()
    #     env.reward(env.control_skel, i)
    # env.reset(0, False)
    # env.reward(env.control_skel, 0)

    # PID Test stuff
    start_frame = 0
    target_frame = 0
    env.sync_skel_to_frame(env.control_skel, target_frame, 0, 0)

    # [print(dof_name, env.metadict[dof_name]) for dof_name in env._actuated_dof_names]

    # print("Provided Target Q: \n", env.control_skel.q[6:])
    target_state = env.gencoordtuple_as_pos_and_eulerlist(env.control_skel)
    pos, vel = target_state
    target_angles = pos[1][1:]
    # print("Provided Target Angles\n", target_angles)

    obs = env.sync_skel_to_frame(env.control_skel, start_frame, 0, 0)

    while True:
        env.framenum = target_frame
        s, r, done, info = env.step(np.concatenate(target_angles))
        env.render()
