from skeleton import Skeleton
from utils3d import *
from xml.dom import minidom
from xml.etree import ElementTree
import argparse
import math
import numpy as np
import os
import re
import xml.etree.ElementTree as ET

CYLINDER_RADIUS = .5

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    # Sourced from https://pymotw.com/2/xml/etree/ElementTree/create.html
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def num2string(num):
    return "{0:.5f}".format(num + 0)

def vec2string(vec):
    """
    Return a space-delineated string of numbers in the vector, with no enclosing
    brackets. Uses num2string internally
    """
    return " ".join([num2string(i) for i in vec])

def bodyname(bone):
    return bone.name + "_body"

def dump_bodies(skeleton, skeleton_xml):
    """
    Given an XML element (an ETElement), dump the skeleton's bone objects
    as the element's children

    This method expects all joint angles in the skeleton to be zero. Else, all
    the axes and angles will be messed up

    It also handles all the calculations concerning axes and joints

    """

    # Ensure all positions are at their "default" values; also, we need to make use
    # of the per-instance attributes which the function adds/sets
    skeleton.update_bone_positions()

    for bone in [skeleton.root] + skeleton.bones:

        body_xml = ET.SubElement(skeleton_xml, "body")
        body_xml.set("name", bodyname(bone))

        tform_text = vec2string(np.append(bone.base_pos, \
                                    rotationMatrixToEulerAngles( \
                                                bone.sum_transform[:3, :3])))
        ET.SubElement(body_xml, "transformation").text = tform_text

        inertia_xml = ET.SubElement(body_xml, "inertia")
        mass_xml = ET.SubElement(inertia_xml, "mass")
        mass_xml.text = str(1)
        ET.SubElement(inertia_xml, "offset").text = "0 0 0"

        def add_cylinder(xml_parent):

            geo_xml = ET.SubElement(xml_parent, "geometry")

            geo_cylinder = ET.SubElement(geo_xml, "cylinder")
            radius = ET.SubElement(geo_cylinder, "radius")
            height = ET.SubElement(geo_cylinder, "height")
            radius.text = num2string(CYLINDER_RADIUS)
            height.text = num2string(bone.length)

        def add_box(xml_parent):
            geo_xml = ET.SubElement(xml_parent, "geometry")
            geo_box = ET.SubElement(geo_xml, "box")
            box_size = ET.SubElement(geo_box, "size")
            # Having bone length along x definitely the right move
            box_size.text = vec2string([bone.length, CYLINDER_RADIUS,
                                        CYLINDER_RADIUS])


        # TODO Figure out how to do things properly
        # TODO Add collision xml back in, the process will be exactly as below
        # Dart doesn't provde functions for putting a bone along an vector, so I
        # construct a transformation bringing the vector to the x-axis, invert it
        # then define the geometry along the x-axis
        vis_xml = ET.SubElement(body_xml, "visualization_shape")
        direction_matrix = rmatrix_v2x(bone.direction)
        rangles = rotationMatrixToEulerAngles(direction_matrix)

        # The coordinate for a box is that of its center instead of an edge, so I move the
        # center out so that the bone is positioned properly
        trans_offset = np.average([bone.base_pos, bone.end_pos], axis=0) - bone.base_pos
        ET.SubElement(vis_xml, "transformation").text = vec2string(trans_offset) + " " + \
                                                        vec2string(rangles)
        add_box(vis_xml)

def write_joint_xml(skeleton_xml, bone):

    joint_xml = ET.SubElement(skeleton_xml, "joint")

    ET.SubElement(joint_xml, "parent").text = bodyname(bone.parent)
    ET.SubElement(joint_xml, "child").text = bodyname(bone)
    # TODO Come up with a better naming scheme if needed
    joint_xml.set("name", bone.name)

    # The joints are by default in the child link frame. This is a problem since
    # the asf/amc data is parent-frame oriented (I think :) so set the angular
    # transformation here The transformation is child frame -> joint frame, so we
    # take (parent frame -> child frame)^-1

    # TODO maybe I use the raw rotation transform instead of the
    # coordinate-augmented one? I dont think so, but it's something to keep in
    # mind

    # TODO Fantastic, both of the below produce the same result... On the other
    # hand, setting positions works now, though that might be entirely
    # unrelated...
    p2c_angular = bone.local_transform[:3, :3]
    # p2c_angular = angular_transform(bone.theta_radians)
    c2p_angular = np.linalg.inv(p2c_angular)
    c2p_angles = rotationMatrixToEulerAngles(c2p_angular)
    ET.SubElement(joint_xml, "transformation").text = "0 0 0 " + vec2string(c2p_angles)

    axes = bone.dofs.replace("r", "").split(" ") if bone.dofs \
            is not None else ""


    jtype = ""
    if len(axes) == 0:
        # TODO turns out that fixed joint type is unsupported, lucky me...
        # I can maybe make a revolute joint with upper and lower limit of 0?
        # raise NotImplementedError("Fixed joint type unsupported" + child.name)
        jtype = "free"
    elif len(axes) == 1:
        jtype = "revolute"
    elif len(axes) == 2:
        jtype = "universal"
    elif len(axes) == 3:
        jtype = "euler"
        ET.SubElement(joint_xml, "axis_order").text = "xyz"
    else:
        raise RuntimeError("Invalid number of axes")

    joint_xml.set("type", jtype)
    for index, axis in enumerate(axes):
        axis_tag = "axis" + ("" if index == 0 else str(index + 1))

        axis_vstr = ""
        if axis == "x":
            axis_vstr = "1 0 0"
        elif axis == "y":
            axis_vstr = "0 1 0"
        elif axis == "z":
            axis_vstr = "0 0 1"

        axis_xml = ET.SubElement(joint_xml, axis_tag)

        ET.SubElement(axis_xml, "xyz").text = axis_vstr
        # TODO Insert this and hope things work
        # TODO I dont think it does anything
        # ET.SubElement(axis_xml, "use_parent_model_frame")
        # TODO implement joint limits!!
        # limit_xml = ET.SubElement(axis_xml, "limit")
        # ET.SubElement(limit_xml, "lower").text = "-3"
        # ET.SubElement(limit_xml, "upper").text = "3"

        # TODO implement dynamics
        # dynamics = ET.SubElement(axis_xml, "dynamics")
        # ET.SubElement(dynamics, "damping").text = "1"
        # ET.SubElement(dynamics, "stiffness").text = "0"


    # TODO Setting positions to 0 makes me a bit nervous as to why it
    # doesn't work...
    # Stuff that shouldnt be required but included just to be safe
    if len(axes) != 0:
        ET.SubElement(joint_xml, "init_pos").text = " ".join(["0"] * len(axes))
        ET.SubElement(joint_xml, "init_vel").text = "0"

def dump_joints(skeleton, skeleton_xml):
    """Given a skeleton object and an xml root, dump joints
    """

    # Setup a special joint for the root
    root_joint = ET.SubElement(skeleton_xml, "joint")
    root_joint.set("name", "world_to_root")
    ET.SubElement(root_joint, "parent").text = "world"
    ET.SubElement(root_joint, "child").text = bodyname(skeleton.root)
    root_joint.set("type", "free")

    for bone in skeleton.bones:

        write_joint_xml(skeleton_xml, bone)

def dump_asf_to_skel(skeleton):

    skeleton_xml = ET.Element("skeleton")
    skeleton_xml.set("name", skeleton.name)
    # TODO Again, fill this up with not-zeroes later
    ET.SubElement(skeleton_xml, "transformation").text = "0 0 0 0 0 0"
    dump_bodies(skeleton, skeleton_xml)
    dump_joints(skeleton, skeleton_xml)

    return "\n".join(prettify(skeleton_xml).splitlines()[1:])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dumps an asf file to a .skel")

    parser.add_argument("--asf", dest="asf_path", default=False)

    args = parser.parse_args()

    skel = Skeleton(args.asf_path)

    new_skel = dump_asf_to_skel(skel)

    start_flag = r"<!--START-->"
    end_flag = r"<!--END-->"
    source_fname = r"test/original/human_box.skel"
    dest_fname = r"test/human.skel"

    with open(source_fname, "r") as f:
        file_text = "".join(f.readlines())

    try:
        os.remove(dest_fname)
    except FileNotFoundError:
        pass

    with open(dest_fname, "w") as f:
        file_text = re.sub(start_flag + ".*" + end_flag,
                           start_flag + "\n" + new_skel + "\n" + end_flag,
                           file_text, flags=re.DOTALL)
        f.write(file_text)
