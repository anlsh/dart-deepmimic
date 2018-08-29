__author__ = 'anish'


from amc import AMC
from asf_skeleton import ASF_Skeleton
from gym import utils
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

# Customizable parameters
ROOT_THETA_KEY = "root_theta"
ROOT_POS_KEY = "root_pos"
REFMOTION_DT = 1 / 120

# ROOT_KEY isn't customizeable. It should correspond
# to the name of the root node in the amc (which is usually "root")
ROOT_KEY = "root"

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
                 max_action_magnitude, default_damping,
                 default_spring,
                 visualize, simsteps_per_dataframe,
                 screen_width,
                 screen_height):

        self.statemode = statemode
        self.actionmode = actionmode
        self.simsteps_per_dataframe = simsteps_per_dataframe
        self.pos_init_noise = pos_init_noise
        self.vel_init_noise = vel_init_noise
        self.max_action_magnitude = max_action_magnitude
        self.default_damping = default_damping
        self.default_spring = default_spring
        self.reward_cutoff = reward_cutoff

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

        self.__control_skeleton_path = control_skeleton_path

        ###########################################################
        # Extract dof info so that states can be converted easily #
        ###########################################################

        world = pydart.World(REFMOTION_DT / self.simsteps_per_dataframe,
                             control_skeleton_path)
        asf = ASF_Skeleton(asf_path)

        self.ref_skel = world.skeletons[-1]
        amc = AMC(reference_motion_path)
        raw_framelist = amc.frames
        self.metadict = get_metadict(raw_framelist[0],
                                     self.ref_skel.dofs, asf)

        self.pos_frames, self.vel_frames = self.convert_frames(raw_framelist)
        self.num_frames = len(self.pos_frames)

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
                          * (len(self.metadict) - 1)
        self.action_limits = [self.max_action_magnitude
                              * np.ones(self.action_dim),
                              -self.max_action_magnitude
                              * np.ones(self.action_dim)]

        self.control_skel = self.ref_skel
        self.load_world()

        ##################################
        # Simulation stuff for DeepMimic #
        ##################################
        self.old_skelq = self.control_skel.q

        self.p_gain = p_gain
        self.d_gain = d_gain

        if self.p_gain < 0 or self.d_gain < 0:
            raise RuntimeError("All PID gains should be positive")

        self.__P = self.p_gain * np.ndarray(self.control_skel.num_dofs())
        self.__D = self.d_gain * np.ndarray(self.control_skel.num_dofs())

    def load_world(self):

        super(DartDeepMimicEnv, self).__init__([self.__control_skeleton_path],
                                               1,
                                               len(self._get_obs()),
                                               self.action_limits,
                                               REFMOTION_DT / self.simsteps_per_dataframe,
                                               "parameter",
                                               "continuous",
                                               self.__visualize,
                                               not self.__visualize)
        self.control_skel = self.dart_world.skeletons[-1]

        # Have to explicitly enable self collisions
        self.control_skel.set_self_collision_check(True)

        # TODO Parse damping from the skel file
        # TODO Add parameters for default damping, spring coefficients
        # TODO dont want to damp the root joint
        for joint in self.control_skel.joints:
            for index in range(joint.num_dofs()):
                joint.set_damping_coefficient(index, self.default_damping)
                joint.set_spring_stiffness(index, self.default_spring)

    def convert_frames(self, raw_framelist):
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

        pos_frames = [None] * len(raw_framelist)
        vel_framelist = [None] * len(raw_framelist)

        for i in range(len(raw_framelist)):
            old_i = i - 1 if i > 0 else 0

            current_frame = raw_framelist[i]
            old_frame = raw_framelist[old_i]

            pos_frame = [None] * len(raw_framelist[i])
            vel_frame = [None] * len(raw_framelist[i])

            # Root data is a little bit special, so we handle it here
            curr_root_data = np.array(current_frame[0][1])
            curr_root_pos, curr_root_theta = curr_root_data[:3], curr_root_data[3:]
            old_root_data = np.array(old_frame[0][1])
            old_root_pos, old_root_theta = old_root_data[:3], old_root_data[3:]
            pos_frame[0] = (ROOT_KEY,
                            np.concatenate([curr_root_pos,
                                            sd2rr(curr_root_theta)]))
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
                dof_indices, order = self.metadict[joint_name]
                start_index, end_index = dof_indices[0], dof_indices[-1]

                curr_theta = expand_angle(curr_joint_angles, order)
                old_theta = expand_angle(old_frame[joint_index][1], order)

                current_rotation_euler = compress_angle(sd2rr(curr_theta), order)
                velocity_rotation_euler = compress_angle(sd2rr(np.subtract(curr_theta,
                                                                           old_theta))) / REFMOTION_DT

                pos_frame[joint_index] = (joint_name, current_rotation_euler)
                vel_frame[joint_index] = (joint_name, velocity_rotation_euler)

            pos_frames[i] = pos_frame
            vel_framelist[i] = vel_frame

        return pos_frames, vel_framelist

    def sync_skel_to_frame(self, skel, frame_index, noise):
        """
        Given a skeleton and mocap frame index, use self.metadict to sync all
        the dofs. Will work on different skeleton objects as long as they're
        created from the same file (like self.control_skel and self.ref_skel)

        If noise is true, then positions and velocities will have random normal
        noise added to them
        """
        pos_frame = self.pos_frames[frame_index]
        vel_frame = self.vel_frames[frame_index]

        def map_dofs(dof_list, pos_list, vel_list, noise):
            """
            Given a list of dof objects, set their positions and velocities
            accordingly. Noise is a boolean
            """
            for dof, pos, vel in zip(dof_list, pos_list, vel_list):
                pos = np.random.normal(pos, self.pos_init_noise if noise else 0)
                vel = np.random.normal(vel, self.vel_init_noise if noise else 0)

                # The float function is required, I've had some strange issues
                # without it
                dof.set_position(float(pos))
                dof.set_velocity(float(vel))

        # World to root joint is a bit special so we handle it here...
        root_pos_data = pos_frame[0][1]
        root_vel_data = vel_frame[0][1]
        map_dofs(skel.dofs[3:6], root_pos_data[:3], root_vel_data[:3], noise)
        # The root pos/vel is never fuzzed (mostly to prevent clipping into the
        # ground)
        map_dofs(skel.dofs[0:3], root_pos_data[3:], root_vel_data[3:], False)

        joint_index = 0
        # And handle the rest of the dofs normally
        for joint_name, joint_angles in pos_frame[1:]:
            joint_index += 1
            dof_indices, order = self.metadict[joint_name]
            start_index, end_index = dof_indices[0], dof_indices[-1]

            joint_velocities = vel_frame[joint_index][1]

            map_dofs(skel.dofs[start_index : end_index + 1],
                     joint_angles, joint_velocities, noise)

    def _get_obs(self):
        """
        Return a 1-dimensional vector of the skeleton's state, as defined by
        the state code
        """

        state = None

        if self.statemode == StateMode.GEN_EULER:
            state = self.gencoordtuple_as_pos_and_eulerlist(
                self.control_skel)
        elif self.statemode == StateMode.GEN_QUAT:
            state = self.gencoordtuple_as_pos_and_qautlist(
                self.control_skel)
        elif self.statemode == StateMode.GEN_AXIS:
            state = self.gencoordtuple_as_pos_and_axisanglelist(
                self.control_skel)
        else:
            raise RuntimeError("Unrecognized or unimpletmented state code: "
                               + str(statemode))

        pos, vel = state
        posvector = np.concatenate([pos[0], np.concatenate(pos[1])])
        velvector = np.concatenate([vel[0], np.concatenate(vel[1])])
        return np.concatenate([posvector, velvector])

    def _genq_to_pos_and_eulerdict(self, generalized_q):
        """
        @type generalized_q: A vector of dof values, as given by skel.q or .dq

        @return: A tuple where
            - index 0 is the root positional component
            - [1] is a dictionary mapping dof names to fully specified euler
              angles in xyz order
        @type: Tuple
        """

        root_translation = generalized_q[3:6]
        expanded_angles = {}
        expanded_angles[ROOT_THETA_KEY] = expand_angle(generalized_q[0:3],
                                                       "xyz")
        for dof_name in self.metadict:
            if dof_name == ROOT_KEY:
                continue
            indices, order = self.metadict[dof_name]
            fi = indices[0]
            li = indices[-1]
            expanded_angles[dof_name] = expand_angle(generalized_q[fi:li],
                                                     order)
        return root_translation, expanded_angles

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

        pos, angles_dict = self._genq_to_pos_and_eulerdict(skeleton.q)
        dpos, dangles_dict = self._genq_to_pos_and_eulerdict(skeleton.dq)

        angles = np.array(list(angles_dict.values()))
        dangles = np.array(list(dangles_dict.values()))

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
        pos_info, vel_info = self.gencoordtuple_as_pos_and_eulerlist(skel)
        pos, angles = pos_info
        dpos, dangles = vel_info

        angles = [axisangle_from_euler(*t, axes="rxyz") for t in angles]
        dangles = [axisangle_from_euler(*t, axes="rxyz") for t in dangles]

        return (pos, angles), (dpos, dangles)

    def _target_full_euler_from_action(self, raw_action):
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
            raise RuntimeError("Unrecognized or unimpletmented action code: "
                               + str(actionmode))

    def reward(self, skel, framenum):

        self.sync_skel_to_frame(self.ref_skel, framenum, False)
        pos, vel = self.gencoordtuple_as_pos_and_qautlist(skel)
        refpos, refvel = self.gencoordtuple_as_pos_and_qautlist(self.ref_skel)

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

        # TODO No quaternion difference used in the paper, but that seems wrong...

        velocity_quats = zip(dangles, drefangles)
        velocity_error = [quaternion_difference(new, old)
                          for new, old in velocity_quats]

        veldiffmag = sum([quaternion_rotation_angle(v)**2
                          for v in velocity_error])

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

        outerweights = [self.pos_weight, self.vel_weight,
                        self.ee_weight, self.com_weight]

        innerweights = [self.pos_inner_weight, self.vel_inner_weight,
                        self.ee_inner_weight, self.com_inner_weight]

        diffmags = [posdiffmag, veldiffmag, eediffmag, comdiffmag]


        reward = sum([ow * exp(iw * diff) for ow, iw, diff in zip(outerweights,
                                                                innerweights,
                                                                diffmags)])

        return(reward)

    def torques_by_pd(self, target_angles, current_angles,
                      past_angles):
        """
        Given target, current, and past angles (all lists of 3-vectors
        representing fully-specified euler angles) of the actuated dofs, return
        torques for the WHOLE SHEBANG.

        This method returns a vector of torques for EVERY DOF in the entire
        skeleton. (even unactuated ones; however, torques on these are always
        zero) This means that it takes care of compressing angles to their
        respective orders, placing them in the right spots in a vector of size
        skel.num_dofs, etc

        The resultant torques are clamped to within action magnitude
        """
        current_error = target_angles - current_angles
        past_error = target_angles - past_angles

        error_rate = (current_error - past_error) / self.dt

        # compression phase
        actuated_dof_names = [key for key in self.metadict
                              if key != ROOT_KEY]
        compressed_current_error = [compress_angle(current_error[i],
                                                  self.metadict[key][1])
                                   for i, key in enumerate(actuated_dof_names)]

        compressed_error_rate = [compress_angle(error_rate[i],
                                               self.metadict[key][1])
                                for i, key in enumerate(actuated_dof_names)]

        expanded_current_error = np.zeros(self.control_skel.num_dofs())
        expanded_error_rate = np.zeros(self.control_skel.num_dofs())

        for index, key in enumerate(actuated_dof_names):
            dof_indices = self.metadict[key][0]
            f, l = dof_indices[0], dof_indices[-1] + 1
            expanded_current_error[f:l] = compressed_current_error[index]

        for index, key in enumerate(actuated_dof_names):
            dof_indices = self.metadict[key][0]
            f, l = dof_indices[0], dof_indices[-1] + 1
            expanded_error_rate[f:l] = compressed_error_rate[index]

        ret = self.p_gain * expanded_current_error + self.d_gain * expanded_error_rate
        ret = np.clip(ret, -self.max_action_magnitude, self.max_action_magnitude)

        return ret

    def step(self, a):

        actuation_targets = self._target_full_euler_from_action(a)

        _, current_euler = self._genq_to_pos_and_eulerdict(self.control_skel.q)
        actuated_angles = np.array([current_euler[key]
                                  for key in current_euler
                                  if key != ROOT_THETA_KEY])

        _, old_euler = self._genq_to_pos_and_eulerdict(self.old_skelq)
        old_actuated_angles = np.array([old_euler[key]
                                        for key in old_euler
                                        if key != ROOT_THETA_KEY])

        torques = self.torques_by_pd(actuation_targets,
                                     actuated_angles,
                                     old_actuated_angles)

        self.old_skelq = self.control_skel.q

        self.do_simulation(torques, self.simsteps_per_dataframe)

        newstate = self._get_obs()
        reward = self.reward(self.control_skel, self.framenum)
        extrainfo = {}
        done = self.framenum == self.num_frames - 1 \
               or (reward < self.reward_cutoff)
        if not np.isfinite(newstate).all():
            raise RuntimeError("Ran into an infinite state, terminating")

        self.framenum += 1

        return newstate, reward, done, extrainfo

    def reset(self, framenum=None, noise=True):

        if framenum is None:
            framenum = random.randint(0, self.num_frames - 1)
        self.framenum = framenum

        self.sync_skel_to_frame(self.control_skel, self.framenum, noise)

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -70
        self._get_viewer().scene.tb.trans[1] = -40

    def render(self, mode='human', close=False):
            # if not self.disableViewer:
        if True:
            self._get_viewer().scene.tb.trans[0] = -self.dart_world.skeletons[-1].com()[0]*1
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

if __name__ == "__main__":

    # Don't run this as main, there's really not too much point

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
    parser.add_argument('--visualize', default=False,
                        help="DOESN'T DO ANYTHING RIGHT NOW: True if you want a window to render to")
    parser.add_argument('--max-action-magnitude', type=float, default=90,
                        help="Maximum torque")
    parser.add_argument('--default-damping', type=float, default=2,
                        help="Default damping coefficient for joints")
    parser.add_argument('--default-spring', type=float, default=0,
                        help="Default spring stiffness for joints")
    parser.add_argument('--simsteps-per-dataframe', type=int, default=10,
                        help="Number of simulation steps per frame of mocap" +
                        " data")
    parser.add_argument('--reward-cutoff', type=float, default=.2,
                        help="Terminate the episode when rewards below this threshold are calculated. Should be in range (0, 1)")
    parser.add_argument('--window-width', type=int, default=80,
                        help="Window width")
    parser.add_argument('--window-height', type=int, default=45,
                        help="Window height")

    parser.add_argument('--pos-init-noise', type=float, default=.05,
                        help="Standard deviation of the position init noise")
    parser.add_argument('--vel-init-noise', type=float, default=.05,
                        help="Standart deviation of the velocity init noise")

    parser.add_argument('--pos-weight', type=float, default=.65,
                        help="Weighting for the pos difference in the reward")
    parser.add_argument('--pos-inner-weight', type=float, default=-2,
                        help="Coefficient for pos difference exponentiation in reward")

    parser.add_argument('--vel-weight', type=float, default=.1,
                        help="Weighting for the pos difference in the reward")
    parser.add_argument('--vel-inner-weight', type=float, default=-.1,
                        help="Coefficient for vel difference exponentiation in reward")

    parser.add_argument('--ee-weight', type=float, default=.15,
                        help="Weighting for the pos difference in the reward")
    parser.add_argument('--ee-inner-weight', type=float, default=-40,
                        help="Coefficient for pos difference exponentiation in reward")

    parser.add_argument('--com-weight', type=float, default=.1,
                        help="Weighting for the com difference in the reward")
    parser.add_argument('--com-inner-weight', type=float, default=-10,
                        help="Coefficient for com difference exponentiation in reward")

    parser.add_argument('--p-gain', type=float, default=300,
                        help="P for the PD controller")
    parser.add_argument('--d-gain', type=float, default=50,
                        help="D for the PD controller")

    args = parser.parse_args()

    env = DartDeepMimicEnv(args.control_skel_path, args.asf_path,
                           args.ref_motion_path,
                           args.state_mode, args.action_mode,
                           args.p_gain, args.d_gain,
                           args.pos_init_noise, args.vel_init_noise,
                           args.reward_cutoff,
                           args.pos_weight, args.pos_inner_weight,
                           args.vel_weight, args.vel_inner_weight,
                           args.ee_weight, args.ee_inner_weight,
                           args.com_weight, args.com_inner_weight,
                           args.max_action_magnitude,
                           args.default_damping, args.default_spring,
                           args.visualize,
                           args.simsteps_per_dataframe,
                           args.window_width, args.window_height)

    env.reset(0, True)
    done = False
    while True:
        env.render()
        a = env.action_space.sample()
        state, reward, done, info = env.step(a)
        if done:
            print("Done, reset")
            env.reset()

    # for i in range(env.num_frames):
    #     env.reset(i, False)
    #     env.render()
    #     env.reward(env.control_skel, i)
    # env.reset(0, False)
    # env.reward(env.control_skel, 0)
