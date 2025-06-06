#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import struct
import sys
import rospy
import subprocess
import math
import csv
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
import baxter_interface
from gazebo_msgs.srv import GetModelState
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
import actionlib
import random

class GripperClient(object):
    def __init__(self, gripper):
        ns = 'robot/end_effector/' + gripper + '_gripper/'
        self._client = actionlib.SimpleActionClient(
            ns + "gripper_action",
            GripperCommandAction,
        )
        self._goal = GripperCommandGoal()

        if not self._client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("Servidor de ação do grip {} não encontrado. Encerrando...".format(gripper))
            rospy.signal_shutdown("Servidor de ação não encontrado")
            sys.exit(1)
        self.clear()

    def command(self, position, effort):
        self._goal.command.position = position
        self._goal.command.max_effort = effort
        self._client.send_goal(self._goal)
        self._client.wait_for_result(rospy.Duration(1.0))        

    def clear(self):
        self._goal = GripperCommandGoal()


def start_gripper_action_server():
    try:
        subprocess.Popen(["rosrun", "baxter_interface", "gripper_action_server.py"])
    except Exception as e:
        rospy.logerr("Falha ao iniciar o servidor de ações do gripper: {}".format(e))
        sys.exit(1)

def move_to_position(limb, joint_angles):
    limb_interface = baxter_interface.Limb(limb)
    print("Moving arm to position: ", joint_angles)
    limb_interface.set_joint_position_speed(0.8)
    limb_interface.move_to_joint_positions(joint_angles, timeout=5)

def angle_to_quaternion(theta):
    rad = math.radians(theta)  # Converter graus para radianos
    x = math.sin(rad / 2)
    y = math.cos(rad / 2)
    return x, y

def execute_sequence(limb, iksvc, ikreq):
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    angles = [45, 90, 135, 180, 225, 270, 315, 360]
    poses = [
        PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(x=0.5, y=-0.4, z=-0.1),
                orientation=Quaternion(x=0.0, y=1, z=0.0, w=0.0),
            ),
        ),
    ]

#    for angle in angles:
#        x, y = angle_to_quaternion(angle)
#        pose = PoseStamped(
#            header=hdr,
#            pose=Pose(
#                position=Point(x=0.43751842625508053, y=-0.21810565906522353, z=0.0),
#                orientation=Quaternion(x=x, y=y, z=0.0, w=0.0),
#           ),
#        )
#        poses.append(pose)

    for pose in poses:
        ikreq.pose_stamp = [pose]
        try:
            rospy.wait_for_service(iksvc.resolved_name, 5.0)
            resp = iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % e)
            return 1

        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        if resp_seeds[0] != resp.RESULT_INVALID:
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            move_to_position(limb, limb_joints)
        else:
            print("INVALID POSE - No Valid Joint Solution Found. (sequence)")
            return 1

    return 0

def normalization (x_normalized, y_normalized):
    x_min, x_max = 0.4189, 0.77258
    y_min, y_max = -0.1221, 0.12347
    
    x = x_normalized * (-x_max + x_min) + x_max
    y = y_normalized * (-y_max + y_min) + y_max
    
    return x, y

def home_position(limb, iksvc, ikreq):
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    poses = [
        PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(x=0.2, y=-0.8, z=0.2),
                orientation=Quaternion(x=0.0, y=1.0, z=0.0, w=0.0),
            ),
        ),
    ]

    for pose in poses:
        ikreq.pose_stamp = [pose]
        try:
            rospy.wait_for_service(iksvc.resolved_name, 5.0)
            resp = iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % e)
            return 1

        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        if resp_seeds[0] != resp.RESULT_INVALID:
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            move_to_position(limb, limb_joints)
        else:
            print("INVALID POSE - No Valid Joint Solution Found. (home_position)")
            return 1

    return 0

def calculate_z(object_name):
    table_height = -0.1734
    offset_gripper = 55 
    a, b = 90, 0.1

    x_pick = None
    y_pick = None
    theta_pick = None
    object_height = None

    with open('objects.csv', 'r') as infile:
        reader = csv.reader(infile)
    
        headers = next(reader)
        height_index = headers.index('height')  
        object_name_index = headers.index('objectName')
        x_index = headers.index('x')  
        y_index = headers.index('y')
        theta_index = headers.index('theta')
        
        for row in reader:
            if row[object_name_index] == object_name:  
                object_height = float(row[height_index])
                x_pick = float(row[x_index])  
                y_pick = float(row[y_index]) 
                theta_pick = float(row[theta_index]) 
                break

    if object_height is None:
        raise ValueError("Object with name " + object_name + " not found in the CSV file.")


    z_object = (b * (object_height - offset_gripper)) / a
    z_object += table_height

    return z_object, x_pick, y_pick, theta_pick

def handle_object(limb, iksvc, ikreq, gc, position):
    gc.command(position=100.0, effort=50.0)
    name, x_place, y_place, theta_place = position
    z_object, x_pick, y_pick, theta_pick = calculate_z(name) 
    x_rot_pick, y_rot_pick = angle_to_quaternion(theta_pick)
    #print('z-height of {} is {}'.format(name, z_object))
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    pose_object = [
        PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(x=x_pick, y=y_pick, z=0.15),
                orientation=Quaternion(x=x_rot_pick, y=y_rot_pick, z=0.0, w=0.0),
            ),
        ),
        PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(x=x_pick, y=y_pick, z=z_object),
                orientation=Quaternion(x=x_rot_pick, y=y_rot_pick, z=0.0, w=0.0),
            ),
        ),
    ]

    for pose in pose_object:
        if not move_with_ik(limb, iksvc, ikreq, pose):
            return 1
    gc.command(position=0.0, effort=50.0)

    x_max = 0.345987
    y_max = 0.227554
    x_normalized, y_normalized = x_place/x_max, y_place/y_max
    x_real, y_real = normalization(x_normalized, y_normalized)
    print("x_real: ", x_real)
    print("y_real: ", y_real)
    x_rot_place, y_rot_place = angle_to_quaternion(theta_place)
    pose_object = [
        PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(x=x_pick, y=y_pick, z=0.15),  
                orientation=Quaternion(x=0.0, y=1.0, z=0.0, w=0.0),
            ),
        ),
        PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(x=x_real,y=y_real,z=0.15),
                orientation=Quaternion(x=x_rot_place, y=y_rot_place, z=0.0, w=0.0),
            ),
        ),
        PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=x_real,  
                    y=y_real,
                    z=z_object,
                ),
                orientation=Quaternion(x=x_rot_place, y=y_rot_place, z=0.0, w=0.0),
            ),
        ),
    ]

    for pose in pose_object:
        if not move_with_ik(limb, iksvc, ikreq, pose):
            return 1
        
    gc.command(position=100.0, effort=50.0)
    rospy.sleep(2)
    pose_object = [
        PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=x_real,  
                    y=y_real,
                    z=0.15,
                ),
                orientation=Quaternion(x=x_rot_place, y=y_rot_place, z=0.0, w=0.0),
            ),
        ),
    ]

    for pose in pose_object:
        if not move_with_ik(limb, iksvc, ikreq, pose):
            return 1

    return 0


def move_with_ik(limb, iksvc, ikreq, pose):
    """
    Auxiliar para calcular e mover o braço para uma pose usando IK.
    """
    ikreq.pose_stamp = [pose]
    try:
        rospy.wait_for_service(iksvc.resolved_name, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr("Falha na chamada ao serviço IK: {}".format(str(e)))
        return False

    resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
    if resp_seeds[0] != resp.RESULT_INVALID:
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        move_to_position(limb, limb_joints)
        return True
    else:
        rospy.logwarn("POSE INVÁLIDA - Sem solução válida encontrada para a posição: x={}, y={}, z={}".format(
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            ))
        return Falsegripper_clients
    

def main():
    parser = argparse.ArgumentParser(description="Baxter RSDK Inverse Kinematics Example")
    parser.add_argument('-l', '--limb', choices=['left', 'right'], required=True, help="the limb to test")
    start_gripper_action_server()
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("rsdk_ik_service_client")
    ns = "ExternalTools/" + args.limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    gc = GripperClient(args.limb)

    gc.command(position=100.0, effort=50.0)
    rospy.sleep(1)
    #home_position(args.limb, iksvc, ikreq)
    execute_sequence(args.limb, iksvc, ikreq)
    positions = [
    ("003 Cracker Box", 0.26626995708596707, 0.035340280332803724, 179.4675064086914),  # (name, x_place, y_place, theta_place)
    ("004 Sugar Box", 0.30039356574118137,0.17987946696138382,93.30286145210266),
    ("008 Pudding Box", 0.04009796851766109,0.05654022221589088,2.9866665601730347),
    ("010 Potted Meat Can", 0.0293622190117836,0.17317922250807286,1.4743191003799438),
    ("007 Tuna Fish Can", 0.1833485800293088,0.1716560124346018,2.8204715251922607),
    ("025 Mug", 0.19138941071355342,0.1653026080582142,6.6002994775772095),
    ("006 Mustard Bottle", 0.1247933078173399,0.05578200799798966,172.24402785301208),
    ("005 Tomato Coup Can", 0.02964034344407916,0.1709995634715557,3.6406105756759644),
    ("017 Orange", 0.17614792785662411,0.16972222845339777,22.028274536132812),
    ("011 Banana", 0.2488943205846548,0.10625318668657542,71.80688202381134),
    ("016 Pear", 0.21902322243511677,0.10925532952117921,81.74250841140747),
    ("012 Strawberry", 0.1437573356052339,0.11411795361542702,80.64979255199432),
    ("014 Lemon", 0.12496468022918701,0.1078711498015523,108.08722972869873),
    ("057 Racquetball", 0.13159567895632981,0.11925751266551018,93.02166938781738),
    
    ]

   # for position in positions:
   #     handle_object(args.limb, iksvc, ikreq, gc, position)
   # home_position(args.limb, iksvc, ikreq)

    return 0

if __name__ == '__main__':
    sys.exit(main())