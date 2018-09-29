from dartdeepmimic import DartDeepMimicEnv
from amc import AMC


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


class AMCDartDeepMimicEnv(DartDeepMimicEnv):

    def __init__(self, *args, **kwargs):
        super(AMCDartDeepMimicEnv, self).__init__(*args, **kwargs)

    def construct_frames(self, ref_skel, ref_motion_path):
        """
        AMC data is given in sequential degrees, while dart specifies angles
        in rotating radians. The conversion is quite expensive, so we precomute
        all positions and velocities and store the results
        """

        raw_framelist = AMC(ref_motion_path).frames

        num_frames = len(raw_framelist)
        elements_per_frame = len(raw_framelist[0])

        pos_frames = [None] * num_frames
        vel_frames = [None] * num_frames
        quat_frames = [None] * num_frames
        com_frames = [None] * num_frames
        ee_frames = [None] * num_frames

        for i in range(len(raw_framelist)):
            old_i = i - 1 if i > 0 else 0

            current_frame = raw_framelist[i]
            old_frame = raw_framelist[old_i]

            q = np.zeros(len(ref_skel.q))
            dq = np.zeros(len(ref_skel.dq))

            # Root data is a little bit special, so we handle it here
            curr_root_data = np.array(current_frame[0][1])
            curr_root_pos, curr_root_theta = \
                                    curr_root_data[:3], curr_root_data[3:]
            old_root_data = np.array(old_frame[0][1])
            old_root_pos, old_root_theta = old_root_data[:3], old_root_data[3:]
            q[3:6] = curr_root_pos
            q[0:3] = sd2rr(curr_root_theta)
            dq[3:6] = np.subtract(curr_root_pos, old_root_pos) / self.refmotion_dt
            dq[0:3] = euler_velocity(curr_root_theta, old_root_theta,
                                     self.refmotion_dt)

            # Deal with the non-root joints in full generality
            joint_index = 0
            for joint_name, curr_joint_angles in current_frame[1:]:
                joint_index += 1
                dof_indices, _ = self.metadict[joint_name]

                length = dof_indices[-1] + 1 - dof_indices[0]

                curr_theta = pad2length(curr_joint_angles, 3)
                old_theta = pad2length(old_frame[joint_index][1], 3)

                # TODO This is not angular velocity at all..
                vel_theta = euler_velocity(curr_theta,
                                           old_theta,
                                           self.refmotion_dt)[:length]
                curr_theta = sd2rr(curr_theta)[:length]

                q[dof_indices[0]:dof_indices[-1] + 1] = curr_theta
                dq[dof_indices[0]:dof_indices[-1] + 1] = vel_theta

            pos_frames[i] = q
            vel_frames[i] = dq

            map_dofs(ref_skel.dofs, q, dq, 0, 0)
            com_frames[i] = ref_skel.com()
            quat_frames[i] = self.quaternion_angles(ref_skel)
            # TODO Parse actual end positions
            ee_frames[i] = [ref_skel.bodynodes[ii].to_world(END_OFFSET)
                            for ii in self._end_effector_indices]

        return num_frames, (pos_frames, vel_frames, quat_frames, com_frames,
                            ee_frames)


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -80
        self._get_viewer().scene.tb.trans[1] = -40
        self._get_viewer().scene.tb.trans[0] = 0
