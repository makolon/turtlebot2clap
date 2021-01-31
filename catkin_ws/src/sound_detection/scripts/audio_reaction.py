#!/usr/bin/env python
import rospy
import numpy as np
import actionlib
import time
from smach import State, StateMachine
import smach
from record import Record
from wav2csv import wav2csv
from music import Music
from move_goal import goal_pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
# from keras.models import load_model
import glob

amplitudes = []
angles = []
quaternions = []

class Recording(State):
    def __init__(self):
        State.__init__(self, outcomes=["success"])

    def execute(self, userdata):
        print("Now I'm recording")
        rospy.sleep(1.0)
        Record()
	return "success"

class Wav2Csv(State):
    def __init__(self):
        State.__init__(self, outcomes=["success"])

    def execute(self, userdata):
        print("Now I'm transfering wav data to csv data")
        rospy.sleep(1.0)
        wav2csv()
	return "success"

class AudioClassification(State):
    def __init__(self):
        State.__init__(self, outcomes=["success", "failure"])

    def execute(self):
        print("Now I'm classificate the input audio data")
        rospy.sleep(1.0)
        model = load_model("./models/esc50._105_0.8096_0.8200.hdf5")
        file_path = sorted(glob.glob("csvfile/*.csv"))
        for f in file_path:
            results = model.predict(f)
            class_id = max_class(results)
            if class_id == 35:
                return 'success'
            else:
                pass

class EstimateAngle(State):
    def __init__(self):
        State.__init__(self, outcomes=["success"])

    def execute(self, userdata):
        print("Now I'm running Music algorithm")
        rospy.sleep(1.0)
        file_path = sorted(glob.glob("csvfile/*.csv"))
        global amplitudes
        global angles
        global quaternions
        for f in file_path:
            music = Music(f)
            R = music.calc_R()
            spec = music.calc_spec(R)
            amplitude, angle = music.max_likelihood_sound(spec)
            quat = music.quat_from_euler(angle)
            amplitudes.append(amplitude)
            angles.append(angle)
            quaternions.append(quat)
            print("Amplitude is {}, and Angle is {}".format(amplitude, angle))
        return "success"

class MoveGoal(State):
    def __init__(self):
        State.__init__(self, outcomes=["success", "failure"])

    def execute(self, userdata):
        print("Now I'm moving to the goal")
        client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        client.wait_for_server()
        direction = 2 # It's not accurate
        while True:
	    position = [distance * np.cos(angles[0]*np.pi/180), distance * np.sin(angles[0]*np.pi/180), 0]
            goal = goal_pose(position, quaternions[0])
            client.send_goal(goal)
            result = client.wait_for_result()
        if result:
            return "success"
        else:
            return "failure"

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

def main():
    rospy.init_node("audio_reaction")
    sm = smach.StateMachine(outcomes=["succeeded"])
    with sm:
        smach.StateMachine.add("Recording", Recording(), transitions={"success":"Wav2Csv"})
        smach.StateMachine.add("Wav2Csv", Wav2Csv(), transitions={"success":"EstimateAngle"})
        smach.StateMachine.add("EstimateAngle", EstimateAngle(), transitions={"success":"MoveGoal"})
        smach.StateMachine.add("MoveGoal", MoveGoal(), transitions={"success":"succeeded", "failure":"Recording"})
    outcome = sm.execute()

if __name__ == "__main__":
    main()
"""
if __name__ == "__main__":
    rospy.init_node("audio_reaction")
    file_path = sorted(glob.glob("csvfile/*.csv"))
    # model = load_model("./models/esc50._105_0.8096_0.8200.hdf5")
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
"""
