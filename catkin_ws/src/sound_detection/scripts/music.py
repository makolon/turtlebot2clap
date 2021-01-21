#!/usr/bin/env python
import glob
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy.linalg as LA
import math
import sys
import os

class Music(object):
    # TODO
    #   - file path is changed!
    #   - adjust the number of self.res, which means resolution.
    #   - adjust csv file in order to load file 5 seconds.
    def __init__(self, csvPath):
        self.data = pd.read_csv(csvPath, engine='python')
        self.data = self.data.drop(columns='Unnamed: 0')
        self.M = 2
        self.n = len(self.data)
        self.res = 16000

    def calc_R(self):
        data = self.data.iloc[0:self.n, :]
        X = signal.hilbert(data.T)
        X = X.T
        R = [[0 for i in range(self.M)] for j in range(self.M)]
        for i in range(self.M):
            for j in range(self.M):
                R_ij = 0
                for k in range(self.n):
                    R_ij += X[k, i] * np.conjugate(X[k, j])
                R[i][j] = R_ij / self.n
        return R

    def calc_spec(self, R):
        W, V = LA.eig(R)
        E = V[:, 1:]
        div = np.pi / self.res
        start = -np.pi / 2
        music_spec = np.empty(self.res)

        for x in range(self.res):
            sin = math.sin(start)
            a = np.empty(self.M, dtype=complex)
            for i in range(self.M):
                s = pow(np.e, (complex(0, -(i+1)*np.pi)*sin))
                a[i] = s
            a_el = a.reshape((1, self.M))
            a_el = np.conjugate(a_el)
            E_el = np.conjugate(E.T)
            numerator = np.dot(a_el, a)
            b_1 = np.dot(a_el, E)
            b_2 = np.dot(b_1, E_el)
            denominator = np.dot(b_2, a)
            music_spec[x] = self.M * (np.log10(numerator.real/denominator.real))
            start += div
        return music_spec

    def max_likelihood_sound(self, music_spec):
        max_num = 0
        count = 0
        angle = np.linspace(-90, 90, self.res)
        for i in range(len(music_spec)):
            if music_spec[i] > max_num:
                max_num = music_spec[i]
                count = i
        music_direction = max_num
        max_angle = angle[count]
        return music_direction, max_angle

    def plot_x(self, music_spec):
        x_axis = np.linspace(-90, 90, self.res)
        plt.plot(x_axis, music_spec)
        plt.xlabel("Angle[angle]")
        plt.ylabel("Amplitude[db]")
        plt.show()

    def quat_from_euler(self, angle):
        """
        q_0 = np.cos(psi/2)*np.cos(theta/2)*np.cos(phi/2)+np.sin(psi/2)*np.sin(theta/2)*np.sin(phi/2)
        q_1 = np.sin(psi/2)*np.cos(theta/2)*np.cos(phi/2)-np.cos(psi/2)*np.sin(theta/2)*np.sin(phi/2)
        q_2 = np.cos(psi/2)*np.sin(theta/2)*np.cos(phi/2)+np.sin(psi/2)*np.cos(theta/2)*np.sin(phi/2)
        q_3 = np.cos(psi/2)*np.cos(theta/2)*np.sin(phi/2)-np.sin(psi/2)*np.sin(theta/2)*np.cos(psi/2)
        """
        angle = angle * np.pi / 360
        q_0 = np.cos(angle)
        q_1 = 0.0
        q_2 = 0.0
        q_3 = np.sin(angle)
        q = [q_0, q_1, q_2, q_3]
        return q

if __name__ == "__main__":
    file_path = sorted(glob.glob("csvfile/*.csv"))
    amplitudes = []
    angles = []
    quaternions = []
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
        # print("Quaternion is {}".format(quat))
    music.plot_x(spec)
