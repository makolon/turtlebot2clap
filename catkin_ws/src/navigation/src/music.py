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
    #   - adjust the number of self.res, which means resolution.
    #   - adjust csv file in order to load file 5 seconds.
    def __init__(self, csvPath):
        self.data = pd.read_csv(csvPath, engine='python')
        self.data = self.data.drop(columns='Unnamed: 0')
        self.M = 2
        self.n = len(self.data)
        self.res = 44100
        self.X = []

    def calc_R(self):
        # print(self.data)
        data = self.data.iloc[0:self.n, :]
        self.X = signal.hilbert(data.T)
        self.X = self.X.T
        R = [[0 for i in range(self.M)] for j in range(self.M)]
        for i in range(self.M):
            for j in range(self.M):
                R_ij = 0
                for k in range(self.n):
		    # print(self.X[k, i])
                    R_ij += self.X[k, i] * np.conjugate(self.X[k, j])
                R[i][j] = R_ij.real / self.n
	        # print(R[i][j])
	print(R)        
	return R

    def calc_spec(self, R):
        W, V = LA.eig(R)
        E = V[:, :]
	print("W is :", W)
	print("V is :", V)
	print("E is :", E)
        div = np.pi / self.res
        start = - np.pi / 2
        music_spec = np.empty(self.res)

        for x in range(self.res):
            sin = math.sin(start)
            a = np.empty(self.M, dtype=complex)
            for i in range(self.M):
                s = pow(np.e, (complex(0, -(i+1)*np.pi)*sin))
                a[i] = s
            a_el = a.reshape((1, self.M))
            a_el = np.conjugate(a_el)
	    print(a_el)
            E_el = np.conjugate(E.T)
            numerator = np.dot(a_el, a)
            b_1 = np.dot(a_el, E)
            b_2 = np.dot(b_1, E_el)
            denominator = np.dot(b_2, a)
	    # print("denominator is :", denominator)
	    print("numerator is :", numerator)
            music_spec[x] = self.M * (np.log10(numerator.real/denominator.real))
            start += div
        return music_spec

    def max_likelihood_sound(self, music_spec):
        max_num = 0
        for i in range(len(music_spec)):
            if music_spec[i] > max_num:
                max_num = music_spec[i]
        music_direction = max_num
        return music_direction

    def plot_x(self, music_spec):
        x_axis = np.linspace(-90, 90, self.res)
        plt.plot(x_axis, music_spec)
        plt.xlabel("Angle[angle]")
        plt.ylabel("Amplitude[db]")
        plt.show()

if __name__ == "__main__":
    file_path = '/home/makolon/turtlebot2clap/catkin_ws/src/navigation/src/output_Output_stereo.csv'
    music = Music(file_path)
    R = music.calc_R()
    spec = music.calc_spec(R)
    direction = music.max_likelihood_sound(spec)
    print(direction)
    # music.plot_x(spec)
