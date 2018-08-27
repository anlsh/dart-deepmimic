from asf_skeleton import ASF_Skeleton
from transformations import euler_from_matrix, compose_matrix
from xml.dom import minidom
from xml.etree import ElementTree
import argparse
import math
import numpy as np
import os
import re
import utils
import xml.etree.ElementTree as ET

CYLINDER_RADIUS = .5

# XML stuff will appear opaque if you haven't read
# https://dartsim.github.io/skel_file_format.html

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    # Sourced from https://pymotw.com/2/xml/etree/ElementTree/create.html
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def num2string(num):
    """
    Function to provide common file-wide number formatting
    """

    # The "+ 0" gets rid of "-0"s being printed
    return "{0:.5f}".format(num + 0)

def vec2string(vec):
    """
    Return a space-delineated string of numbers in the vector, with no enclosing
    brackets. Uses num2string internally
    """
    return " ".join([num2string(i) for i in vec])

def bodyname(joint):
    """
    Returns the name of a joint
    """
    return joint.name + "_body"

def add_cylinder(xml_parent, length):

    geo_xml = ET.SubElement(xml_parent, "geometry")

    geo_cylinder = ET.SubElement(geo_xml, "cylinder")
    radius = ET.SubElement(geo_cylinder, "radius")
    height = ET.SubElement(geo_cylinder, "height")
    radius.text = num2string(CYLINDER_RADIUS)
    height.text = num2string(length)

def add_box(xml_parent, length):
    geo_xml = ET.SubElement(xml_parent, "geometry")
    geo_box = ET.SubElement(geo_xml, "box")
    box_size = ET.SubElement(geo_box, "size")
    box_size.text = vec2string([length, 2 * CYLINDER_RADIUS, CYLINDER_RADIUS])

def add_capsule(xml_parent, length):
    geo_xml = ET.SubElement(xml_parent, "geometry")
    geo_cap = ET.SubElement(geo_xml, "capsule")
    ET.SubElement(geo_cap, "height").text = num2string(length)
    ET.SubElement(geo_cap, "radius").text = num2string(2 * CYLINDER_RADIUS)
    # box_size.text = vec2string([length, 2 * CYLINDER_RADIUS, CYLINDER_RADIUS])

def dump_bodies(asf_skeleton, skeleton_xml):
    """
    Given an XML element (an ETElement), dump the skeleton's joint objects
    as the element's children

    This method expects all joint angles in the skeleton to be zero. Else, all
    the axes and angles will be messed up

    It also handles all the calculations concerning axes and joints
    """

    # Ensure all positions are at their "default" values; if the skeleton's pose
    # is different from that in the ASF file, everything will be off :|
    asf_skeleton.update_joint_positions()

    for joint in [asf_skeleton.root] + asf_skeleton.joints:

        body_xml = ET.SubElement(skeleton_xml, "body")
        body_xml.set("name", bodyname(joint))

        ################################
        # POSITION AND COORDINATE AXES #
        ################################

        rmatrix = joint.ctrans
        tform_text = vec2string(np.append(joint.base_pos,
                                          euler_from_matrix(rmatrix[:3, :3],
                                                            axes="rxyz")))
        ET.SubElement(body_xml, "transformation").text = tform_text

        ########################################
        # VISUALIZATION AND COLLISION GEOMETRY #
        ########################################

        # Direction vectors and axes are specified wrt to global reference
        # frame in asf files (and thus in joints), so we construct a
        # transformation to the local reference frame (as dart expects it)
        local_direction = np.matmul(joint.ctrans_inv[:3, :3], joint.direction)
        direction_matrix = utils.rmatrix_x2v(local_direction)
        rangles = utils.rotationMatrixToEulerAngles(direction_matrix)
        trans_offset = joint.length * local_direction / 2
        tform_vector = np.append(trans_offset, rangles)

        for shape in ["visualization", "collision"]:
            shape_xml = ET.SubElement(body_xml, shape + "_shape")
            ET.SubElement(shape_xml, "transformation").text = \
                                                        vec2string(tform_vector)
            add_box(shape_xml, joint.length)

        ###################
        # INERTIA SECTION #
        ###################

        inertia_xml = ET.SubElement(body_xml, "inertia")
        mass_xml = ET.SubElement(inertia_xml, "mass")
        mass_xml.text = str(1)
        ET.SubElement(inertia_xml, "offset").text = vec2string(trans_offset)

def write_joint_xml(skeleton_xml, joint):

    joint_xml = ET.SubElement(skeleton_xml, "joint")

    ET.SubElement(joint_xml, "parent").text = bodyname(joint.parent)
    ET.SubElement(joint_xml, "child").text = bodyname(joint)
    joint_xml.set("name", joint.name)

    ET.SubElement(joint_xml, "transformation").text = "0 0 0 0 0 0"

    jtype = ""
    if len(joint.dofs) == 0:
        # TODO Why oh why does this kill things
        joint_xml.set("type", "weld")
        return
    elif len(joint.dofs) == 1:
        jtype = "revolute"
    elif len(joint.dofs) == 2:
        jtype = "universal"
    elif len(joint.dofs) == 3:
        jtype = "euler"
        ET.SubElement(joint_xml, "axis_order").text = "xyz"
    else:
        raise RuntimeError("Invalid number of axes")

    joint_xml.set("type", jtype)
    for index, axis in enumerate(joint.dofs):
        axis_tag = "axis" + ("" if index == 0 else str(index + 1))

        axis_vstr = ""
        if axis == "x":
            li = 0
            axis_vstr = "1 0 0"
        elif axis == "y":
            li = 1
            axis_vstr = "0 1 0"
        elif axis == "z":
            li = 2
            axis_vstr = "0 0 1"

        axis_xml = ET.SubElement(joint_xml, axis_tag)

        ET.SubElement(axis_xml, "xyz").text = axis_vstr

        if joint.limits[li] is not None:
            limit_xml = ET.SubElement(axis_xml, "limit")
            low, high = joint.limits[li]
            ET.SubElement(limit_xml, "lower").text = num2string(low)
            ET.SubElement(limit_xml, "upper").text = num2string(high)

        dynamics = ET.SubElement(axis_xml, "dynamics")
        ET.SubElement(dynamics, "damping").text = "1"
        ET.SubElement(dynamics, "friction").text = "0"
        ET.SubElement(dynamics, "spring_rest_position").text = "0"
        ET.SubElement(dynamics, "spring_stiffness").text = "0"

def dump_joints(asf_skeleton, skeleton_xml):
    """
    Given a skeleton object and an xml root, dump joints
    """

    # Set up a special joint for the root
    root_joint = ET.SubElement(skeleton_xml, "joint")
    root_joint.set("name", "root")
    ET.SubElement(root_joint, "parent").text = "world"
    ET.SubElement(root_joint, "child").text = bodyname(asf_skeleton.root)
    root_joint.set("type", "free")

    for joint in asf_skeleton.joints:

        write_joint_xml(skeleton_xml, joint)

def dump_asf_to_skel(asf_skeleton):

    skeleton_xml = ET.Element("skeleton")
    skeleton_xml.set("name", asf_skeleton.name)
    ET.SubElement(skeleton_xml, "transformation").text = "0 0 0 0 0 0"
    dump_bodies(asf_skeleton, skeleton_xml)
    dump_joints(asf_skeleton, skeleton_xml)

    # The first line is always <xml_version 1.0>, so skip that
    return "\n".join(prettify(skeleton_xml).splitlines()[1:])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dumps an asf file to a .skel")

    parser.add_argument("--asf", dest="asf_path", required=True)
    parser.add_argument("--dest", dest="dest_path", required=True)

    rsrc_help = "File to replace text in; if unspecified, uses dest. " \
                + "If it doesn't exist, it will be created"
    parser.add_argument("--rsrc", dest="rsrc", required=False, help=rsrc_help)

    replace_help = "If true, then attempt to find <!--START--> " \
    + "and <!--END--> tags in the destination file and replace whatever is in" \
    + "between with the generated xml"

    parser.add_argument("--replace", dest="replace", help=replace_help,
                        default=False)

    args = parser.parse_args()

    skel = ASF_Skeleton(args.asf_path)

    skel_xml = dump_asf_to_skel(skel)

    start_flag = r"<!--START-->"
    end_flag = r"<!--END-->"

    if args.rsrc is None:
        args.rsrc = args.dest_path

    with open(args.rsrc, "r") as f:
        file_text = "".join(f.readlines())

    try:
        os.remove(args.dest_path)
    except FileNotFoundError:
        pass

    with open(args.dest_path, "w") as f:
        if args.replace:
            file_text = re.sub(start_flag + ".*" + end_flag,
                               start_flag + "\n" + skel_xml + "\n" + end_flag,
                               file_text, flags=re.DOTALL)
        else:
            file_text = skel_xml

        f.write(file_text)
