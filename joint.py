import numpy as np
import math
from utils import to_radians, from_radians

from transformations import compose_matrix

def expand_angle(in_angle, order="xyz", initial_element=0):
    """
    Given an array of 1-3 elements and an order like 'xy' 'z' or something,
    return a tuple (theta_x, theta_y, theta_z)

    in_angle and order should have same number of elements

    Examples:
    angle([1,2], "xz") -> (1, 0, 2)
    angle([1,2,3], "yzx") -> (3, 1, 2)
    """
    blank = [initial_element] * 3
    index_map = {"x": 0, "y": 1, "z": 2}
    for axis, val in zip(order, in_angle):
        blank[index_map[axis]] = val

    return blank

def compress_angle(in_angle, order="xyz"):
    """
    Given a vector of length 3, extract the relevant vector components
    """
    index_map = {"x": 0, "y": 1, "z": 2}
    ret = np.array([0] * len(order))
    for index, code in enumerate(order):
        # print("For index " + str(index) + " extract the " + str(in_angle[index_map[code]]))
        # print("Before set, ret is " + str(ret))
        print(str(ret[index]) + " is being set " + str(in_angle[index_map[code]]))
        print(type(ret[index:index+1]))
        ret[index] = in_angle[index_map[code]]
        print("After set, ret is " + str(ret))

    print("Compress got an input of " + str(in_angle) + " " + str(order)
          + "\n ----> " + str(ret))

    return ret

class Joint:

    def from_dict(dictionary):
        """
        Meant to take in dictionaries formatted like cgikit does and return
        a Joint object
        """
        id_ = int(dictionary["id"][0])
        name = dictionary["name"][0]
        direction = np.array([float(i) for i in dictionary["direction"]])
        axis_degrees = np.array([float(i) for i in dictionary["axis"][:-1]])
        length = float(dictionary["length"][0])
        theta = np.array([0, 0, 0])
        parent = None

        dofs = " ".join(dictionary["dof"]) if "dof" in dictionary else ""
        dofs = dofs.replace("r", "").replace(" ", "")

        limits = [None] * 3
        if "limits" in dictionary:
            zipped = dictionary["limits"]
            for i, lim in enumerate(zipped):
                zipped[i] = to_radians(lim)

            # expand_angle is useful for more than just angles, but this is the
            # only place it's used in generality so far so there's not much
            # reason to rename it
            limits = expand_angle(zipped, dofs, None)

        return Joint(id_, name, direction, to_radians(axis_degrees),
                     length, dofs, parent, limits)


    def __init__(self, id_, name, direction, axis, length, dofs, parent=None,
                 limits=[None] * 3):

        self.id_ = id_
        self.name = name
        self.direction = direction
        self.length = length
        self.dofs = dofs

        self.limits = limits

        self._axis = [0, 0, 0]
        self.axis_radians = axis
        self._theta = [0, 0, 0]

        self._parent = None
        self.parent = parent

        self.__update_ctrans()
        self.__update_ttrans

    def __update_ctrans(self):
        self.ctrans = compose_matrix(angles=self.axis_radians,
                                     translate=[0,0,0])
        self.ctrans_inv = np.linalg.inv(self.ctrans)

    def __update_ttrans(self):
        self.ttrans = compose_matrix(translate=self.parent.offset)
        self.ttrans_inv = np.linalg.inv(self.ttrans)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, new):
        self._parent = new
        if self.parent is not None:
            self.__update_ttrans()

    @property
    def axis_degrees(self):
        return from_radians(self._axis)

    @axis_degrees.setter
    def axis_degrees(self, new_axis):
        self._axis = to_radians(new_axis)
        self.__update_ctrans()

    @property
    def axis_radians(self):
        return self._axis

    @axis_radians.setter
    def axis_radians(self, new):
        self._axis = new
        self.__update_ctrans()

    @property
    def theta_degrees(self):
        return from_radians(self._theta)

    @theta_degrees.setter
    def theta_degrees(self, new_theta):
        self._theta = to_radians(new_theta)

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

        return np.matmul(self.ttrans,
                              np.matmul(self.ctrans,
                                    np.matmul(compose_matrix(angles=
                                                             self.theta_radians),
                                              self.ctrans_inv)))
