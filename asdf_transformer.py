import numpy as np
import math
import re
import time

from skeleton import Skeleton

import pydart2 as pydart

# # Calculates Rotation Matrix given euler angles.
# def eulerAnglesToRotationMatrix(theta) :

#     # Sourced from https://www.learnopencv.com/rotation-matrix-to-euler-angles/

#     R_x = np.array([[1,         0,                  0                   ],
#                     [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
#                     [0,         math.sin(theta[0]), math.cos(theta[0])  ]
#                     ])



#     R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
#                     [0,                     1,      0                   ],
#                     [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
#                     ])

#     R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
#                     [math.sin(theta[2]),    math.cos(theta[2]),     0],
#                     [0,                     0,                      1]
#                     ])


#     R = np.dot(R_z, np.dot( R_y, R_x ))

#     return R

# def get_transform_matrix(theta, translation):

#     dimension = len(translation)
#     matrix = np.zeros([dimension + 1] * 2)
#     matrix[:,dimension] = np.append(translation, [1])
#     matrix[:dimension, :dimension] = eulerAnglesToRotationMatrix(theta)

#     return matrix

# class Body:

#     def __init__(self, id_, name, direction, axis, length, dofs, parent=None):

#         self.id_ = id_
#         self.name = name
#         self.direction = np.array(direction)
#         self.axis = np.array(axis)
#         self.length = length
#         self.dofs = dofs
#         self.theta = np.array([0,0,0])

#         self.parent = parent

#         self.position = None

#         def set_rx(tx):
#             self.theta = np.array([tx, 0, 0])

#         def set_ry(ty):
#             self.theta = np.array([0, ty, 0])

#         def set_rz(tz):
#             self.theta = np.array([0, 0, tz])

#         def set_rxy(tx, ty):
#             self.theta = np.array([tx, ty, 0])

#         def set_rxz(tx, tz):
#             self.theta = np.array([tx, 0, tz])

#         def set_ryz(ty, tz):
#             self.theta = np.array([0, ty, tz])

#         def set_rxyz(tx, ty, tz):
#             self.theta = np.array([tx, ty, tz])

#         def set_invalid(theta):
#             raise RuntimeError("Can't set angles on this joint!")

#         if dofs == None:
#             self.set_angles = set_invalid
#         elif dofs == "rx":
#             self.set_angles = set_rx
#         elif dofs == "ry":
#             self.set_angles = set_ry
#         elif dofs == "rz":
#             self.set_angles = set_rz
#         elif dofs == "rx ry":
#             self.set_angles = set_rxy
#         elif dofs == "ry rz":
#             self.set_angles = set_ryz
#         elif dofs == "rx rz":
#             self.set_angles = set_rxz
#         elif dofs == "rx ry rz":
#             self.set_angles = set_rxyz
#         else:
#             raise RuntimeError("Invalid dofs: " + str(dofs))


#     @property
#     def transform(self):
#         return get_transform_matrix(self.theta, self.length * self.direction)

# def parse_body_string(body_string):

#     id_re = re.compile(r"id ([0-9]+)")
#     name_re = re.compile(r"name (.+)")
#     dir_re = re.compile(r"direction ([\.\-+e0-9]+) ([\.\-+e0-9]+) ([\.\-+e0-9]+)")
#     axis_re = re.compile(r"axis ([\.\-+e0-9]+) ([\.\-+e0-9]+) ([\.\-+e0-9]+)")
#     length_re = re.compile(r"length ([\.\-+e0-9]+)")
#     dofs_re = re.compile(r"dof (.+)")

#     id_ = int((id_re.search(body_string).group(1)))
#     name = name_re.search(body_string).group(1)
#     direction = [float(dir_re.search(body_string).group(n)) for n in range(1, 4)]
#     axis = [float(axis_re.search(body_string).group(n)) for n in range(1, 4)]
#     length = float(length_re.search(body_string).group(1))
#     dofs = dofs_re.search(body_string)
#     if dofs is not None:
#         dofs = dofs.group(1)

#     return Body(id_, name, direction, axis, length, dofs)

# def get_bodies_from_file(filename):
#     with open(filename) as f:
#         f_string = "".join(f.readlines())

#     # print(f_string)
#     all_bones_re = re.compile(r":bonedata(.+):hierarchy", re.DOTALL)
#     all_bone_string = all_bones_re.search(f_string).group(1)

#     bone_re = re.compile(r"begin(.+?)end", re.DOTALL)
#     bone_string_matches = bone_re.finditer(all_bone_string)

#     bones = [parse_body_string(match.group(1)) for match in bone_string_matches]


#     name_to_bone = {}
#     for b in bones:
#         name_to_bone[b.name] = b

#     bones.insert(0, Body(-1, "root", [0, 0, 0], [0, 0, 0], 1, "rx ry rz"))
#     name_to_bone["root"] = bones[0]

#     parent_section_re = re.compile(r":hierarchy.*begin(.+)end", re.DOTALL)
#     parent_section = parent_section_re.search(f_string).group(1).strip()

#     for line in parent_section.split("\n"):
#         bone_deps = line.strip().split(" ")
#         for b in bone_deps[1:]:
#             name_to_bone[b].parent = name_to_bone[bone_deps[0]]

#     return bones, name_to_bone

def get_angles_from_amc(filename):
    with open(filename) as f:
        file_contents = "".join(f.readlines())

    frame_strings = re.split(r"^[0-9]+$", file_contents, flags=re.MULTILINE)
    print(len(frame_strings))
    frame_strings = frame_strings[1:]
    # print(frame_strings[0])

    for i, glob in enumerate(frame_strings):
        lines = glob.split("\n")
        lines = [line.split(" ") for line in lines]
        frame_strings[i] = [[line[0]] + [float(l) for l in line[1:]] for line in lines if line != ""]

    return frame_strings

class DotWorld(pydart.World):

    def __init__(self, skeleton, amc):

        self.skeleton = skeleton
        self.amc = amc
        self.count = 0
        pydart.World.__init__(self, 1.0 / 2000.0)

    def render_with_ri(self, ri):
        self.count += 1

        for line in self.amc[self.count % len(self.amc)]:
            if line[0] != "root" and line[0] != "":

                angles = (math.pi / 180 * r for r in line[1:])
                self.skeleton.bone_dict[line[0]].set_theta(*angles)

        self.skeleton.update_bone_positions()

        for bone in self.skeleton.bones:
            if bone.name == "lfemur":
                ri.set_color(0.0, 1.0, 0.0)
            else:
                ri.set_color(1.0, 0.0, 0.0)

            ri.render_sphere(.03 * bone.pos, .03)

if __name__ == "__main__":

    s = Skeleton("/home/anish/Downloads/07.asf")
    amc = get_angles_from_amc("/home/anish/Downloads/07_02.amc")

    pydart.init()
    world = DotWorld(s, amc)
    pydart.gui.viewer.launch(world)
