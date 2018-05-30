from cgkit.asfamc import ASFReader
import numpy as np
import math
from utils3d import get_transform_matrix

class Bone:

    def from_dict(dictionary):
        """
        Meant to take in dictionaries formatted like cgikit does and return a Bone object
        """
        id_ = int(dictionary["id"][0])
        name = dictionary["name"][0]
        direction = np.array([float(i) for i in dictionary["direction"]])
        axis = np.array([float(i) for i in dictionary["axis"][:-1]])
        length = float(dictionary["length"][0])
        dofs = " ".join(dictionary["dof"]) if "dof" in dictionary else None
        theta = np.array([0, 0, 0])
        parent = None

        return Bone(id_, name, direction, axis, length, dofs, parent)


    def __init__(self, id_, name, direction, axis, length, dofs, parent=None):

        self.id_ = id_
        self.name = name
        self.direction = np.array(direction)
        self.axis = np.array(axis)
        self.length = length
        self.dofs = dofs
        self.theta = np.array([0,0,0])

        self.parent = parent

        def set_rx(tx):
            self.theta = np.array([tx, 0, 0])

        def set_ry(ty):
            self.theta = np.array([0, ty, 0])

        def set_rz(tz):
            self.theta = np.array([0, 0, tz])

        def set_rxy(tx, ty):
            self.theta = np.array([tx, ty, 0])

        def set_rxz(tx, tz):
            self.theta = np.array([tx, 0, tz])

        def set_ryz(ty, tz):
            self.theta = np.array([0, ty, tz])

        def set_rxyz(tx, ty, tz):
            self.theta = np.array([tx, ty, tz])

        def set_invalid(theta):
            raise RuntimeError("Can't set angles on this joint!")

        if dofs == None:
            self.set_theta = set_invalid
        elif dofs == "rx":
            self.set_theta = set_rx
        elif dofs == "ry":
            self.set_theta = set_ry
        elif dofs == "rz":
            self.set_theta = set_rz
        elif dofs == "rx ry":
            self.set_theta = set_rxy
        elif dofs == "ry rz":
            self.set_theta = set_ryz
        elif dofs == "rx rz":
            self.set_theta = set_rxz
        elif dofs == "rx ry rz":
            self.set_theta = set_rxyz
        else:
            raise RuntimeError("Invalid dofs: " + str(dofs))

    @property
    def local_transform(self):
        return get_transform_matrix(self.theta,
                                    self.length * self.direction)

class Skeleton:

    def __init__(self, filename):

        self.name = None
        self.bones = None
        self.root = None
        self.units = None
        self.hierarchy = None
        self.bone_dict = {}

        self.root_pos = None
        self.root_theta = None

        reader = ASFReader(filename)
        def __init_name(name):
            self.name = name
        def __init_units(units):
            self.units = units
        def __init_bones(bone_data):
            self.bones = [Bone.from_dict(entry) for entry in bone_data]
            for bone in self.bones:
                self.bone_dict[bone.name] = bone
        def __init_root(root_data):
            self.root_pos = np.array([float(i) for i in root_data["position"]])
            self.root_theta = np.array([float(i) for i in root_data["orientation"]])
        def __init_hierarchy(hierarchy):
            for h in hierarchy:
                parent_name, dep_names = h
                for dep_name in dep_names:
                    if parent_name != "root":
                        self.bone_dict[dep_name].parent = self.bone_dict[parent_name]

        reader.onName = __init_name
        reader.onUnits = __init_units
        reader.onRoot = __init_root
        reader.read()

        bone_reader = ASFReader(filename)
        bone_reader.onBonedata = __init_bones
        bone_reader.onHierarchy = __init_hierarchy
        bone_reader.read()

    def update_bone_positions(self):

        root_transform = get_transform_matrix(self.root_theta, self.root_pos)
        for bone in self.bones:
            bone.global_transform = None
        for bone in self.bones:
            bone.global_transform = np.matmul(bone.parent.global_transform
                                              if bone.parent else root_transform,
                                              bone.local_transform)
            bone.pos = np.matmul(bone.global_transform, np.array([0,0,0,1]))[:-1]
