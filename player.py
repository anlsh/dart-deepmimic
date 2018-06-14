from amc import ASF_AMC
from skeleton import Skeleton
import argparse
import pydart2 as pydart
import numpy as np

from transformations import *

class DotWorld(pydart.World):

    def __init__(self, skeleton, amc, scale=1):

        self.skeleton = skeleton
        self.amc = amc
        self.count = 0
        self.scale = scale
        pydart.World.__init__(self, 1.0 / 2000.0)

    def render_with_ri(self, ri):

        self.count += 1
        if self.amc:
            self.amc.sync_angles(self.count % self.amc.num_frames)

        self.skeleton.update_bone_positions()

        for bone in self.skeleton.bones:

            # if bone.name == "rfemur":
            #     ri.set_color(0.0, 0.0, 0.0)

            if False and bone.name in ["lhumerus", "lfemur", "ltibia", "lfoot"]:
                ri.set_color(1, 0.0, 0)
                cy_length = 10
                cy_base = np.matmul(bone.ctrans, [0, 0, 0, 1])[:3] + bone.base_pos
                cy_end = np.matmul(bone.ctrans, [cy_length, 0, 0, 1])[:3] + bone.base_pos
                ri.render_cylinder_two_points(self.scale * cy_base,
                                              self.scale * cy_end,
                                              3 * self.scale / 30)

                ri.set_color(0, 1.0, 0)
                cy_length = 10
                cy_base = np.matmul(bone.ctrans, [0, 0, 0, 1])[:3] + bone.base_pos
                cy_end = np.matmul(bone.ctrans, [0, cy_length, 0, 1])[:3] + bone.base_pos
                ri.render_cylinder_two_points(self.scale * cy_base,
                                              self.scale * cy_end,
                                              3 * self.scale / 30)
                ri.set_color(0, 0.0, 1)
                cy_length = 10
                cy_base = np.matmul(bone.ctrans, [0, 0, 0, 1])[:3] + bone.base_pos
                cy_end = np.matmul(bone.ctrans, [0, 0, cy_length, 1])[:3] + bone.base_pos
                ri.render_cylinder_two_points(self.scale * cy_base,
                                              self.scale * cy_end,
                                              3 * self.scale / 30)


            ri.set_color(0.0, 0, 0)
            ri.render_sphere(self.scale * bone.base_pos, 10 * self.scale / 20)
            ri.render_cylinder_two_points(self.scale * bone.base_pos,
                                            self.scale * bone.end_pos,
                                            3 * self.scale / 30)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ASF/AMC Movie player')

    parser.add_argument("--asf", dest="asf")
    parser.add_argument("--amc", dest="amc", default=None)
    parser.add_argument("--scale", default=1, type=float)

    args = parser.parse_args()

    s = Skeleton(args.asf)
    amc = ASF_AMC(args.amc, s) if args.amc else None

    sq = s.name2bone
    # sq["lhumerus"].set_theta_degrees(90, 90, 0)

    pydart.init()
    world = DotWorld(s, amc, args.scale)
    pydart.gui.viewer.launch(world)
