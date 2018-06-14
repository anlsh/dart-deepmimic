import numpy as np
import math

from transformations import compose_matrix

def expand_angle(in_angle, order="xyz"):
    """
    Given an array of 1-3 elements and an order like 'xy' 'z' or something,
    return a tuple (theta_x, theta_y, theta_z)

    in_angle and order should have same number of elements

    Examples:
    angle([1,2], "xz") -> (1, 0, 2)
    angle([1,2,3], "yzx") -> (3, 1, 2)
    """
    blank = [0, 0, 0]
    index_map = {"x": 0, "y": 1, "z": 2}
    for axis, val in zip(order, in_angle):
        blank[index_map[axis]] = val

    return blank

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
        dofs = " ".join(dictionary["dof"]) if "dof" in dictionary else ""
        theta = np.array([0, 0, 0])
        parent = None

        return Joint(id_, name, direction, axis_degrees, length, dofs, parent)


    def __init__(self, id_, name, direction, axis, length, dofs, parent=None):

        self.id_ = id_
        self.name = name
        self.direction = direction
        self.length = length
        self.dofs = dofs

        self._axis = None
        self.axis_degrees = axis

        self._theta = None

        self._parent = None
        self.parent = parent

        self.__update_ctrans()
        self.ttrans = compose_matrix()
        self.ttrans_inv = np.linalg.inv(self.ttrans)

        self.dofs = dofs.replace("r", "").replace(" ", "")

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
        return np.multiply(180 / math.pi, self._axis)

    @axis_degrees.setter
    def axis_degrees(self, new_axis):
        self._axis = np.multiply(math.pi / 180, new_axis)
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
    def rtrans(self):
        return compose_matrix(angles=self.theta_radians)

    @property
    def local_transform(self):

        return np.matmul(self.ttrans,
                              np.matmul(self.ctrans,
                                    np.matmul(self.rtrans,
                                              self.ctrans_inv)))
