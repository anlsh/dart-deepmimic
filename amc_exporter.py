"""
Export a .amc (acclaim motion capture) to a .dmc (dart motion capture),
which is the same in every way except all angles are given as rotating
radians instead of sequential degrees
"""

from cgkit.asfamc import AMCReader
from joint import expand_angle
from transformations import compose_matrix, euler_from_matrix
from asf_skeleton import ASF_Skeleton

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dumps an amc file to a dmc")

    parser.add_argument("--asf", dest="asf_path", required=True)
    parser.add_argument("--amc", dest="amc_path", required=True)
    parser.add_argument("--dmc", dest="dmc_path", required=True)

    args = parser.parse_args()

    skel = ASF_Skeleton(args.asf_path)
