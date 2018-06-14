
from cgkit.asfamc import AMCReader
from joint import expand_angle
from skeleton import Skeleton
from transformations import compose_matrix, euler_from_matrix
import math
import numpy as np

class AMC:
    """
    Parent class representing information from a .amc file
    """

    def __init__(self, amc_filename, skeleton):

        self.name = None
        self.frames = []
        self.skeleton = skeleton

        def __init_frame(framenum, data):
            self.frames.append(data)

        reader = AMCReader(amc_filename)
        reader.onFrame = __init_frame
        reader.read()

    def sync_angles(self, framenum):
        """
        Call this method to set all of the skeleton's angles to the values in the
        corresponding frame of the AMC (where 0 is the first frame)
        """
        raise NotImplementedError("Abstract class, use either ASF or SKEL AMC classes")

class ASF_AMC(AMC):

    def sync_angles(self, framenum):

        frame = self.frames[framenum]
        root_data = frame[0][1]
        self.skeleton.root.direction = np.array(root_data[0:3])
        self.skeleton.root.theta_degrees = np.array(root_data[3:])

        for joint_name, joint_data in frame[1:]:
            joint = self.skeleton.name2joint[joint_name].theta_degrees
            joint.theta_degrees = expand_angle(joint_data, joint.dofs)

def sequential_to_rotating_radians(rvector):

    rmatrix = compose_matrix(angles=rvector, angle_order="sxyz")
    return euler_from_matrix(rmatrix[:3, :3], axes="rxyz")


class Skel_AMC(AMC):
    """
    A class to sync AMC frames with a Dart Skeleton object
    """

    def __init__(self, amc_filename, skeleton, skeleton_filename):
        super(Skel_AMC, self).__init__(amc_filename, skeleton)

        # Set up a map of joint names to dof indices
        # start index and window length tuple
        self.joint2window = {}
        self.asf_skeleton = Skeleton(skeleton_filename)

        dof_names = [dof.name for dof in self.skeleton.dofs]

        for joint in self.skeleton.joints:
            i = 0
            while True:
                if skeleton.dofs[i].name[:len(joint.name)] == joint.name:
                    self.joint2window[joint.name] = (i, joint.num_dofs())
                    break
                i += 1

    def sync_angles(self, framenum):

        # framenum = 0

        frame = self.frames[framenum]
        root_data = frame[0][1]

        def zip_dofs(dof_list, pos_list):

            for dof, pos in zip(dof_list, pos_list):
                dof.set_position(pos)

        zip_dofs(self.skeleton.dofs[0:3],
                 sequential_to_rotating_radians(np.multiply(math.pi / 180,
                                                            root_data[3:])))
        zip_dofs(self.skeleton.dofs[3:6], root_data[:3])

        for joint_name, joint_data in frame[1:]:
            index, length = self.joint2window[joint_name]

            # I need this to take advantage of the auto-angle placement
            asf_joint = self.asf_skeleton.name2joint[joint_name]
            asf_joint.theta_degrees = expand_angle(joint_data, asf_joint.dofs)
            rotation_euler = sequential_to_rotating_radians(asf_joint.theta_radians)

            zip_dofs(self.skeleton.dofs[index : index + length],
                     rotation_euler)
