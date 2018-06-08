# Copyright (c) 2015, Disney Research
# All rights reserved.
#
# Author(s): Sehoon Ha <sehoon.ha@disneyresearch.com>
# Disney Research Robotics Group
import sys
from pydart2.gui.pyqt5.window import PyQt5Window
from pydart2.gui.trackball import Trackball
import pydart2 as pydart
import argparse
import math

skel = None

def getViewer(sim, title=None):

	win = PyQt5Window(sim, title)
	win.scene.add_camera(Trackball(theta=0, phi = 0, zoom=1.2,trans=[0,0.0,-30]), 'Hopper_camera')
	win.scene.set_camera(win.scene.num_cameras()-1)
	#win.run()
	return win

class HelpWorld(pydart.World):

    def __init__(self, *args):

        pydart.World.__init__(self, *args)

    def render_with_ri(self, ri):
        ri.render_axes([0,0,0], 5, r_base_ = .2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Views a skel file")

    parser.add_argument("--skel", dest="skel_path", required=True)

    args = parser.parse_args()

    pydart.init(verbose=True)
    print('pydart initialization OK')

    world = HelpWorld(0.0002, args.skel_path)
    skel = world.skeletons[1]

    print(skel.dofs)


    # femur: 12
    # humerus: 52, 64
    # radius: 55, 67
    # tibia: 15, 28

    skel.dofs[55].set_position(math.pi / 2)
    # print(type(skel.dofs[0]))
    # print(skel.joints)
    print('pydart create_world OK')

    window = getViewer(world)
    window.run()
