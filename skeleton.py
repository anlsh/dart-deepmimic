from cgkit.asfamc import ASFReader
from bone import Bone
import numpy as np
from utils3d import get_transform_matrix

class Skeleton:

    def __init__(self, filename):

        self.name = None
        self.bones = None
        self.root = None
        self.units = None
        self.name2bone = {}

        self.root = Bone(-1, "root", [0,0,0], [0,0,0], 1, "rx ry rz")
        self.name2bone["root"] = self.root

        def __init_name(name):
            self.name = name

        def __init_units(units):
            self.units = units

        def __init_bones(bone_data):

            self.bones = [Bone.from_dict(entry) for entry in bone_data]
            for bone in self.bones:
                self.name2bone[bone.name] = bone

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
                    self.name2bone[dep_name].parent = self.name2bone[parent_name]

        reader = ASFReader(filename)
        reader.onName = __init_name
        reader.onUnits = __init_units
        reader.onRoot = __init_root
        reader.onBonedata = __init_bones
        reader.read()

        reader = ASFReader(filename)
        reader.onHierarchy = __init_hierarchy
        reader.read()

    def update_bone_positions(self):

        self.root.sum_transform, _ = get_transform_matrix(self.root.theta_radians,
                                                          self.root.direction)
        self.root.base_pos = self.root.offset
        self.root.end_pos = self.root.offset


        for bone in self.bones:

            bone.sum_transform = np.matmul(bone.parent.sum_transform,
                                           bone.local_transform)

            bone.base_pos = np.matmul(bone.sum_transform,
                                      np.array([0,0,0,1]))[:-1]
            bone.end_pos = np.matmul(bone.sum_transform,
                                     np.append(bone.offset, 1))[:-1]
