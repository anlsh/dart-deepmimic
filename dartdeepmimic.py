__author__ = 'anish'


from amc import AMC
from asf_skeleton import ASF_Skeleton
from gym import utils
from gym.envs.dart import dart_env
from joint import expand_angle, compress_angle
from math import exp, pi
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
    # TODO README EMERGENCY!!!
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
                 pos_weight, pos_inner_weight,
                 vel_weight, vel_inner_weight,
                 ee_weight, ee_inner_weight,
                 com_weight, com_inner_weight,
                 max_action_magnitude, default_damping,
                 default_spring,
                 visualize, frame_skip,
                 screen_width,
                 screen_height):

        self.statemode = statemode
        self.actionmode = actionmode
        self.frame_skip = frame_skip
        self.pos_init_noise = pos_init_noise
        self.vel_init_noise = vel_init_noise
        self.max_action_magnitude = max_action_magnitude
        self.default_damping = default_damping
        self.default_spring = default_spring

        self.framenum = 0

        # TODO I don't know if this parameter actually does anything
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

        # TODO idk what this does actually
        self.calc_dt = REFMOTION_DT / self.frame_skip

        world = pydart.World(self.calc_dt, control_skeleton_path)
        asf = ASF_Skeleton(asf_path)

        self.ref_skel = world.skeletons[1]
        amc = AMC(reference_motion_path)
        self.framelist = amc.frames
        self.metadict = get_metadict(self.framelist[0],
                                     self.ref_skel.dofs, asf)
        self.convert_frames()

        # Setting control skel to ref skel is just a workaround:
        # it's set to its correct value later on
        self.control_skel = self.ref_skel

        self.__end_effector_indices = [i for i, node
                                       in enumerate(self.control_skel.bodynodes)
                                     if len(node.child_bodynodes) == 0]

        self.__end_effector_offsets = [asf.name2joint[
            self.control_skel.bodynodes[i].name[:-5]].offset
                                       for i in self.__end_effector_indices]

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

        #### BEGIN TEST
        # TESTING WORLD LOAD
        super(DartDeepMimicEnv, self).__init__([self.__control_skeleton_path],
                                  self.frame_skip,
                                  len(self._get_obs()),
                                  self.action_limits, self.calc_dt, "parameter",
                                  "continuous", self.__visualize,
                                  not self.__visualize)
        self.load_world()

        #### END TEST

        ##################################
        # Simulation stuff for DeepMimic #
        ##################################
        self.old_skelq = self.control_skel.q

        self.p_gain = p_gain
        self.d_gain = d_gain

        self.__P = self.p_gain * np.ndarray(self.control_skel.num_dofs())
        self.__D = self.d_gain * np.ndarray(self.control_skel.num_dofs())

    def load_world(self):
        self.control_skel = self.dart_world.skeletons[1]

        # Have to explicitly enable self collisions
        self.control_skel.set_self_collision_check(True)

        # TODO Parse damping from the skel file
        # TODO Add parameters for default damping, spring coefficients
        for joint in self.control_skel.joints:
            for index in range(joint.num_dofs()):
                joint.set_damping_coefficient(index, self.default_damping)
                joint.set_spring_stiffness(index, self.default_spring)


    def convert_frames(self):
        """
        AMC data is given in sequential degrees, while dart specifies angles
        in rotating radians. The conversion is quite expensive, so we precomute
        here (destructively modifying the self.frames variable)
        """

        def sequential_degrees_to_rotating_radians(rvector):

            rvector = np.multiply(rvector, pi / 180)

            rmatrix = compose_matrix(angles=rvector, angle_order="sxyz")
            return euler_from_matrix(rmatrix[:3, :3], axes="rxyz")

        for frame in self.framelist:

            # Root is a bit special since it contains translational + angular
            # information, deal with that here
            root_data = frame[0][1]
            root_pos, root_theta = root_data[:3], root_data[3:]
            root_theta = sequential_degrees_to_rotating_radians(root_data[3:])
            frame[0] = (ROOT_KEY, np.concatenate([root_pos, root_theta]))

            index = 0
            for joint_name, joint_angles in frame[1:]:
                index += 1
                dof_indices, order = self.metadict[joint_name]
                start_index, end_index = dof_indices[0], dof_indices[-1]

                theta = expand_angle(joint_angles, order)
                rotation_euler = sequential_degrees_to_rotating_radians(theta)
                new_rotation_euler = compress_angle(rotation_euler, order)
                frame[index] = (joint_name, new_rotation_euler)

    def sync_skel_to_frame(self, skel, frame_index, noise=True):
        """
        Given a skeleton and mocap frame index, use self.metadict to sync all
        the dofs
        """
        frame = self.framelist[frame_index % len(self.framelist)]
        old_frame = self.framelist[(frame_index - 1 if frame_index > 0 else 0)
                                    % len(self.framelist)]

        def map_dofs(dof_list, pos_list, vel_list, noise):
            """
            Noise is a boolean
            """
            for dof, pos, vel in zip(dof_list, pos_list, vel_list):
                pos = np.random.normal(pos, self.pos_init_noise if noise else 0)
                vel = np.random.normal(vel, self.vel_init_noise if noise else 0)
                dof.set_position(float(pos))
                dof.set_velocity(float(vel))

        # World to root joint is a bit special so we handle it here...
        root_data = frame[0][1]
        old_root_data = frame[0][1]
        root_vel = (root_data - old_root_data) / REFMOTION_DT
        map_dofs(skel.dofs[3:6], root_data[:3], root_vel[:3], noise)
        map_dofs(skel.dofs[0:3], root_data[3:], root_vel[3:], False)

        i = 0
        # And handle the rest of the dofs normally
        for joint_name, joint_angles in frame[1:]:
            i += 1
            dof_indices, order = self.metadict[joint_name]
            start_index, end_index = dof_indices[0], dof_indices[-1]

            old_joint_angles = old_frame[i][1]
            joint_velocities = (joint_angles - old_joint_angles) / REFMOTION_DT

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

    def genq_to_pos_and_eulerdict(self, generalized_q):
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

        pos, angles_dict = self.genq_to_pos_and_eulerdict(skeleton.q)
        dpos, dangles_dict = self.genq_to_pos_and_eulerdict(skeleton.dq)

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
        construct from it a set of targets for the ACTUATED degrees of freedom
        (ie the ones in metadict, minus the root)

        Because of how action_dim is defined up in __init__, raw_action
        should always have the correct dimensions
        """

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

    def reward(self, skel, frame):

        pos, vel = self.gencoordtuple_as_pos_and_qautlist(skel)
        pos, angles = pos
        dpos, dangles = vel

        self.sync_skel_to_frame(self.ref_skel, frame - 1
                                if frame > 0 else 0)
        refpos_old, _ = self.gencoordtuple_as_pos_and_qautlist(self.ref_skel)
        refpos_old, refangles_old = refpos_old

        self.sync_skel_to_frame(self.ref_skel, frame)
        refpos, _ = self.gencoordtuple_as_pos_and_qautlist(self.ref_skel)
        refpos, refangles = refpos
        refcom = self.ref_skel.com()

        #####################
        # POSITIONAL REWARD #
        #####################

        posdiff = [quaternion_difference(ra, a)
                   for a, ra in zip(angles, refangles)]
        posdiffmag = sum([norm(d) for d in posdiff])

        ###################
        # VELOCITY REWARD #
        ###################

        # TODO No quaternion difference used in the paper, but that seems wrong...

        data_velocity = [quaternion_difference(new, old)
                         / REFMOTION_DT for new, old in zip(refangles,
                                                            refangles_old)]

        vdiff = [quaternion_difference(s, data) for s, data in zip(dangles,
                                                                   data_velocity)]
        veldiffmag = sum([norm(v) for v in vdiff])

        #######################
        # END EFFECTOR REWARD #
        #######################

        # TODO THe units are off, the paper specifically specifies units of meters

        eediffmag = sum([norm(self.control_skel.bodynodes[i].to_world(offset)
                                        - self.ref_skel.bodynodes[i].to_world(offset))
                         for i, offset in zip(self.__end_effector_indices,
                                              self.__end_effector_offsets)])

        #########################
        # CENTER OF MASS REWARD #
        #########################

        comdiffmag = norm(self.control_skel.com() - refcom)

        ################
        # TOTAL REWARD #
        ################

        outerweights = [self.pos_weight, self.vel_weight,
                        self.ee_weight, self.com_weight]

        innerweights = [self.pos_inner_weight, self.vel_inner_weight,
                        self.ee_inner_weight, self.com_inner_weight]

        diffmags = [posdiffmag, veldiffmag, eediffmag, comdiffmag]

        return sum([ow * exp(iw * diff) for ow, iw, diff in zip(outerweights,
                                                                innerweights,
                                                                diffmags)])

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
        #     self.tau[6:] = self.__PID()

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

    def torques_by_pd(self, target_angles, current_angles,
                      past_angles):
        """
        Given target, current, and past angles (all lists of 3-vectors
        representing fully-specified euler angles) of the actuated dofs, return
        torques for the WHOLE SHEBANG.

        This method returns a vector of torques for EVERY DOF in the entire
        skeleton. This means that it takes care of compressing angles to their
        respective orders, placing them in the right spots in a vector of size
        skel.num_dofs, etc

        Non actuated dofs will of course have torques of 0
        Also it clamps
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

        ret = self.p_gain * expanded_current_error - self.d_gain * expanded_error_rate
        ret = np.clip(ret, -self.max_action_magnitude, self.max_action_magnitude)

        return ret

    def step(self, a):

        actuation_targets = self._target_full_euler_from_action(a)

        _, current_euler = self.genq_to_pos_and_eulerdict(self.control_skel.q)
        actuated_angles = np.array([current_euler[key]
                                  for key in current_euler
                                  if key != ROOT_THETA_KEY])

        _, old_euler = self.genq_to_pos_and_eulerdict(self.old_skelq)
        old_actuated_angles = np.array([old_euler[key]
                                        for key in old_euler
                                        if key != ROOT_THETA_KEY])

        torques = self.torques_by_pd(actuation_targets,
                                     actuated_angles,
                                     old_actuated_angles)

        self.old_skelq = self.control_skel.q

        # Also what is the difference between world step
        self.control_skel.set_forces(torques)
        self.dart_world.step()

        # TODO EMERGENCY DONT DOUBLE COUNT FRAME SKIP
        self.do_simulation(torques, self.frame_skip)

        newstate = self._get_obs()
        reward = self.reward(self.control_skel, self.framenum)
        # TODO Implement more early terminateion stuff
        done = self.framenum == len(self.framelist) - 1 \
               or (not np.isfinite(newstate).all())
        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 200).all() and# (abs(L_angle - self.foot_angles[self.count]) < 10) and (abs(R_angle - self.foot_angles[self.count]) < 10) and
        extrainfo = {}

        self.framenum += 1

        return newstate, reward, done, extrainfo

    def reset(self, framenum=None, noise=True):

        if framenum is None:
            framenum = random.randint(0, len(self.framelist))
        self.framenum = framenum

        self.sync_skel_to_frame(self.control_skel, self.framenum, noise)

        return self._get_obs()

    # def viewer_setup(self):
    #     if not self.disableViewer:
    #         self._get_viewer().scene.tb.trans[0] = 5.0
    #         self._get_viewer().scene.tb.trans[2] = -30
    #         self._get_viewer().scene.tb.trans[1] = 0.0

    def render(self, mode='human', close=False):
            # if not self.disableViewer:
        if True:
            self._get_viewer().scene.tb.trans[0] = -self.dart_world.skeletons[self.track_skeleton_id].com()[0]*1
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
                        help="True if you want a window to render to")
    parser.add_argument('--max-action-magnitude', type=float, default=90,
                        help="Maximum torque")
    parser.add_argument('--default-damping', type=float, default=2,
                        help="Default damping coefficient for joints")
    parser.add_argument('--default-spring', type=float, default=0,
                        help="Default spring stiffness for joints")
    parser.add_argument('--frame-skip', type=int, default=1,
                        help="Number of simulation steps per frame of mocap" +
                        " data")
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
    print(args.vel_init_noise, "The vel init noise")

    env = DartDeepMimicEnv(args.control_skel_path, args.asf_path,
                           args.ref_motion_path,
                           args.state_mode, args.action_mode,
                           args.p_gain, args.d_gain,
                           args.pos_init_noise, args.vel_init_noise,
                           args.pos_weight, args.pos_inner_weight,
                           args.vel_weight, args.vel_inner_weight,
                           args.ee_weight, args.ee_inner_weight,
                           args.com_weight, args.com_inner_weight,
                           args.max_action_magnitude,
                           args.default_damping, args.default_spring,
                           args.visualize,
                           args.frame_skip,
                           args.window_width, args.window_height)

    env.reset(0, True)
    # env.reset()
    for i in range(1200):
        env.render()
        a = env.action_space.sample()
        state, action, reward, done = env.step(a)
        if done:
            env.reset(0, True)
