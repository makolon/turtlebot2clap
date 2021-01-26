#!/usr/bin/env python
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

def goal_pose(position, orientation):
    goal_pose = MoveBaseGoal()
    goal_pose.target_pose.header.frame_id = 'map'
    goal_pose.target_pose.pose.position.x = position[0]
    goal_pose.target_pose.pose.position.y = position[1]
    goal_pose.target_pose.pose.position.z = position[2]
    goal_pose.target_pose.pose.orientation.x = orientation[0]
    goal_pose.target_pose.pose.orientation.y = orientation[1]
    goal_pose.target_pose.pose.orientation.z = orientation[2]
    goal_pose.target_pose.pose.orientation.w = orientation[3]

    return goal_pose

if __name__ == "__main__":
    rospy.init_node("move_goal")
    client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    client.wait_for_server()
    while True:
        goal = goal_pose(pose, orn)
        client.send_goal(goal)
        client.wait_for_result()
