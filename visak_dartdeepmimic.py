from dartdeepmimic import DartDeepMimicEnv, map_dofs, END_OFFSET
import ddm_argparse
import numpy as np
import argparse
import runner
from copy import copy

class VisakDartDeepMimicEnv(DartDeepMimicEnv):

    def construct_frames(self, ref_motion_path):
        """
        AMC data is given in sequential degrees, while dart specifies angles
        in rotating radians. The conversion is quite expensive, so we precomute
        all positions and velocities and store the results
        """

        raw_framelist = None

        with open("assets/mocap/JustJumpPositions_corrected.txt","rb") as fp:
            WalkPositions = np.loadtxt(fp)
        with open("assets/mocap/JustJumpVelocities_corrected.txt","rb") as fp:
            WalkVelocities = np.loadtxt(fp)

        num_frames = len(WalkPositions)

        pos_frames = [None] * num_frames
        vel_frames = [None] * num_frames
        quat_frames = [None] * num_frames
        com_frames = [None] * num_frames
        ee_frames = [None] * num_frames


        for i in range(len(WalkPositions)):

            updated_pos = WalkPositions[i,:].copy()
            updated_pos[3:6] = updated_pos[3:6][::-1]
            temp = updated_pos[3:6].copy()
            updated_pos[3:6] = updated_pos[0:3]
            updated_pos[0:3] = temp

            updated_vel = WalkVelocities[i,:].copy()
            updated_vel[3:6] = updated_vel[3:6][::-1]
            temp = updated_vel[3:6].copy()
            updated_vel[3:6] = updated_pos[0:3]
            updated_vel[0:3] = temp

            map_dofs(self.ref_skel.dofs, updated_pos,
                     updated_vel, 0, 0)
            pos_frames[i] = updated_pos
            vel_frames[i] = updated_vel
            com_frames[i] = self.ref_skel.com()
            quat_frames[i] = self.quaternion_angles(self.ref_skel)
            # TODO Parse actual end positions
            ee_frames[i] = [self.ref_skel.bodynodes[ii].to_world(END_OFFSET)
                            for ii in self._end_effector_indices]

        return num_frames, (pos_frames, vel_frames, quat_frames, com_frames,
                            ee_frames)

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = 5.0
            self._get_viewer().scene.tb.trans[2] = -7.5
            self._get_viewer().scene.tb.trans[1] = 0.0
            #-10.0

