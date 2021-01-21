#!/usr/bin/env python
import rospy
import numpy as np
import actionlib
from music import Music
from move_goal import goal_pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from keras.models import load_model

def max_class(results):
    max_prob = 0
    for i in range(len(results[0])):
        if max_prob > results[0][i]:
            max_prob = results[0][i]
            max_class = i
    return max_class

def threshold_class(results):
    threshold = 0.6
    for i in range(len(results)):
        if results[i][35] > threshold:
            max_class = 35
    return max_class

if __name__ == "__main__":
    rospy.init_node("audio_reaction")
    file_path = sorted(glob.glob("csvfile/*.csv"))
    model = load_model("./models/esc50._105_0.8096_0.8200.hdf5")
    for f in file_path:
        results = model.predict(f)
        class_id = max_class(results)
        if class_id == 35:
            music = Music()
            R = music.calc_R()
            spec = music.calc_spec(R)
            amplitude, angle = music.max_likelihood_sound(spec)
            orientation = music.quat_from_euler(angle)
            distance = 5 # Is it accurate?
            position = [distance * np.cos(angle*np.pi/180), distance * np.sin(angle*np.pi/180), 0]
            client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
            client.wait_for_server()
            while True:
                goal = goal_pose(position, orientation)
                client.send_goal(goal)
                client.wait_for_result()
            break
        else:
            continue

