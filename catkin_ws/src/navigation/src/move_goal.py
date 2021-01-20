#!/usr/bin/env python
import rospy
from actionlib
from smach import State, StateMachine
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class Waypoint(State):
    def __init__(self, position, orientation):
        State.__init__(self, outcomes=['success'])
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = 'map'
        self.goal.target_pose.pose.position.x = position[0]
        self.goal.target_pose.pose.positoin.y = position[1]
        self.goal.target_pose.pose.position.z = 0.0
        self.goal.target_pose.pose.orientation.x = orientation[0]
        self.goal.target_pose.pose.orientatoin.y = orientation[1]
        self.goal.target_pose.pose.orientation.z = orientation[2]
        self.goal.target_pose.pose.orientation.w = orientation[3]

    def excecute(self, userdata):
        self.client.send_goal(self.goal)
        self.client.wait_for_resulut()
        return 'success'

if __name__ == "__main__":
    patrol = StateMachine('success')
    StateMachine.add(goal[0], Waypoint(goal[1], goal[2]), transitions={'success':waypoints[(i+1)%len(goal)][0]})
    patrol.execute()
