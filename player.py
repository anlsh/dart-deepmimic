from amc import AMC
from skeleton import Skeleton
import argparse
import pydart2 as pydart

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
            self.amc.sync_angles(self.count % self.amc.num_frames,
                                 self.skeleton)

        self.skeleton.update_bone_positions()

        for bone in self.skeleton.bones:

            ri.set_color(1.0, 0, 0)

            # if bone.name == "rfemur":
            #     ri.set_color(0.0, 0.0, 0.0)

            ri.render_sphere(self.scale * bone.base_pos, 10 * self.scale / 20)
            ri.render_cylinder_two_points(self.scale * bone.base_pos,
                                            self.scale * bone.end_pos,
                                            3 * self.scale / 30)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ASF/AMC Movie player')

    parser.add_argument("--skel", dest="skel")
    parser.add_argument("--mov", dest="mov", default=None)
    parser.add_argument("--scale", default=1, type=float)

    args = parser.parse_args()

    s = Skeleton(args.skel)
    amc = AMC(args.mov) if args.mov else None

    pydart.init()
    world = DotWorld(s, amc, args.scale)
    pydart.gui.viewer.launch(world)
