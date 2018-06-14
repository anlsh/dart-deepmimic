from joint import Joint
from cgkit.asfamc import ASFReader
from transformations import compose_matrix
import numpy as np

class Skeleton:

    def __init__(self, filename):

        self.name = None
        self.joints = None
        self.root = None
        self.units = None
        self.name2joint = {}

        self.root = Joint(-1, "root", [0,0,0], [0,0,0], 1, "rx ry rz")
        self.name2joint["root"] = self.root

        def __init_name(name):
            self.name = name

        def __init_units(units):
            self.units = units

        def __init_joints(joint_data):

            self.joints = [Joint.from_dict(entry) for entry in joint_data]
            for joint in self.joints:
                self.name2joint[joint.name] = joint

        def __init_root(root_data):
            self.root.direction = np.array([float(i)
                                            for i in root_data["position"]])
            self.root.theta_degrees = np.array([float(i)
                                                for i in
                                                root_data["orientation"]])

        def __init_hierarchy(hierarchy):
            for h in hierarchy:
                parent_name, dep_names = h
                for dep_name in dep_names:
                    self.name2joint[dep_name].parent = self.name2joint[parent_name]

        reader = ASFReader(filename)
        reader.onName = __init_name
        reader.onUnits = __init_units
        reader.onRoot = __init_root
        reader.onBonedata = __init_joints
        reader.read()

        reader = ASFReader(filename)
        reader.onHierarchy = __init_hierarchy
        reader.read()

    def update_joint_positions(self):

        self.root.sum_transform = compose_matrix(angles=self.root.theta_radians,
                                                 translate=self.root.direction)
        self.root.base_pos = self.root.offset
        self.root.end_pos = self.root.offset
        self.root.sum_ctrans = self.root.ctrans


        for joint in self.joints:

            joint.sum_transform = np.matmul(joint.parent.sum_transform,
                                           joint.local_transform)

            joint.base_pos = np.matmul(joint.sum_transform,
                                      np.array([0,0,0,1]))[:-1]
            joint.end_pos = np.matmul(joint.sum_transform,
                                     np.append(joint.offset, 1))[:-1]
