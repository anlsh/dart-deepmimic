# Author: Anish Moorthy

import sys
from pydart2.gui.pyqt5.window import PyQt5Window
from pydart2.gui.trackball import Trackball
import pydart2 as pydart
import argparse
import math

from amc import Skel_AMC

def getViewer(sim, title=None):

	win = PyQt5Window(sim, title)
	win.scene.add_camera(Trackball(theta=0, phi = 0, zoom=1.2,trans=[0,0.0,-30]), 'Hopper_camera')
	win.scene.set_camera(win.scene.num_cameras()-1)
	#win.run()
	return win

class MovieWorld(pydart.World):

    def __init__(self, *args):

        pydart.World.__init__(self, *args)
        self.amc = None
        self.count = 0

    def set_amc(self, amc):
        self.amc = amc

    def render_with_ri(self, ri):
        self.count += 1
        ri.render_axes([0,0,0], 5, r_base_ = .2)

        if self.amc is not None:
            self.amc.sync_angles(self.count % len(self.amc.frames))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Views a skel file")

    parser.add_argument("--skel", dest="skel_path", required=True)
    parser.add_argument("--asf", dest="asf_path", required=True)
    parser.add_argument("--amc", dest="amc_path", required=False, default=None)

    args = parser.parse_args()

    pydart.init(verbose=True)
    print('pydart initialization OK')

    world = MovieWorld(0.0002, args.skel_path)

    skel = world.skeletons[1]
    amc = Skel_AMC(args.amc_path, skel, args.asf_path) if args.amc_path is not None else None
    world.set_amc(amc)

    print('pydart create_world OK')

    window = getViewer(world)
    window.run()
