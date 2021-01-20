#!/usr/bin/env python
import rospy
import numpy as np
import actionlib
from music import Music
from audio_classifier import Audio_Classifer
from move_goal import Waypoint
from smach import State, StateMachine
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

if __name__ == "__main__":
    rospy.init_node("audio_reaction")
    file_path = sorted(glob.glob("csvfile/*.csv"))
    for f in file_path:
        audio_class = Audio_Classifier()
        class_id = audio_class.predict(f)
        if class_id == 35:
            music = Music()
            R = music.calc_R()
            spec = music.calc_spec(R)
            amplitude, angle = music.max_likelihood_sound(spec)
            quat = music.quat_from_euler(angle)
            distance = 5 # Is it accurate?
            position = [distance * np.cos(angle*np.pi/180), distance * np.sin(angle*np.pi/180), 0]
            patrol = StateMachine('success')
            StateMachine.add('goal', WayPoint(position, quat), transitions={'success':waypoints[(i+1)%len(goal)}[0])
            patrol.execute()
            
            break
        else:
            continue

