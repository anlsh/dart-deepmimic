
from cgkit.asfamc import AMCReader
import numpy as np
import math
from utils3d import *

class AMC:

    def __init__(self, filename):

        self.name = None
        self.frames = []
        self.num_frames = 0

        def __init_frame(framenum, data):
            self.frames.append(data)
            self.num_frames += 1

        reader = AMCReader(filename)
        reader.onFrame = __init_frame
        reader.read()

    def sync_angles(self, framenum, skeleton):
        frame = self.frames[framenum]
        root_data = frame[0][1]
        skeleton.root.direction = np.array(root_data[0:3])
        skeleton.root.theta_degrees = np.array(root_data[3:])

        for bone_name, bone_data in frame[1:]:
            skeleton.name2bone[bone_name].set_theta(*bone_data)
