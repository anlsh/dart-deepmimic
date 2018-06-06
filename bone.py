import numpy as np
import math
from utils3d import *

class Bone:

    def from_dict(dictionary):
        """
        Meant to take in dictionaries formatted like cgikit does and return
        a Bone object
        """
        id_ = int(dictionary["id"][0])
        name = dictionary["name"][0]
        direction = np.array([float(i) for i in dictionary["direction"]])
        axis_degrees = np.array([float(i) for i in dictionary["axis"][:-1]])
        length = float(dictionary["length"][0])
        dofs = " ".join(dictionary["dof"]) if "dof" in dictionary else None
        theta = np.array([0, 0, 0])
        parent = None

        return Bone(id_, name, direction, axis_degrees, length, dofs, parent)


    def __init__(self, id_, name, direction, axis, length, dofs, parent=None):

        self.id_ = id_
        self.name = name
        self.direction = direction
        self.length = length
        self.dofs = dofs

        self._axis = None
        self.axis_degrees = axis

        self._theta = None
        self.theta_degrees = [0, 0, 0]

        self._parent = None
        self.parent = parent

        self.ctrans, self.ctrans_inv = get_transform_matrix(self.axis_radians,
                                                            [0,0,0])
        self.ttrans, self.ttrans_inv = get_transform_matrix([0,0,0], [0,0,0])

        # TODO All of these need to be updated to form "set_theta_radians"
        def set_rx(tx):
            self.theta_degrees = np.array([tx, 0, 0])

        def set_ry(ty):
            self.theta_degrees = np.array([0, ty, 0])

        def set_rz(tz):
            self.theta_degrees = np.array([0, 0, tz])

        def set_rxy(tx, ty):
            self.theta_degrees = np.array([tx, ty, 0])

        def set_rxz(tx, tz):
            self.theta_degrees = np.array([tx, 0, tz])

        def set_ryz(ty, tz):
            self.theta_degrees = np.array([0, ty, tz])

        def set_rxyz(tx, ty, tz):
            self.theta_degrees = np.array([tx, ty, tz])

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
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, new):
        self._parent = new
        if self.parent is not None:
            self.ttrans, self.ttrans_inv = get_transform_matrix([0, 0, 0],
                                                                self.parent.offset)

    @property
    def axis_degrees(self):
        return np.multiply(180 / math.pi, self._axis)

    @axis_degrees.setter
    def axis_degrees(self, new_axis):
        self._axis = np.multiply(math.pi / 180, new_axis)
        self.__update_ctransform()

    @property
    def axis_radians(self):
        return self._axis

    @axis_radians.setter
    def axis_radians(self, new):
        self._axis = new
        self.__update_ctransform()

    def __update_ctransform(self):
        self.ctrans, self.ctrans_inv = get_transform_matrix(self.axis_radians,
                                                                    [0,0,0])

    @property
    def theta_degrees(self):
        return np.multiply(180 / math.pi, self.theta)

    @theta_degrees.setter
    def theta_degrees(self, new_theta):
        self._theta = np.multiply(math.pi / 180, new_theta)

    @property
    def theta_radians(self):
        return self._theta

    @theta_radians.setter
    def theta_radians(self, new):
        self._theta = new

    @property
    def offset(self):
        return self.length * self.direction

    @property
    def local_transform(self):

        rtrans, _ = get_transform_matrix(self._theta,
                                         np.array([0,0,0]))
        return np.matmul(self.ttrans,
                              np.matmul(self.ctrans,
                                    np.matmul(rtrans,
                                              self.ctrans_inv)))
