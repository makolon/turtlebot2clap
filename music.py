import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy.linalg as LA
import math

class Music(object):
    def __init__(self):
        self.data = pd.real_csv("csvPath")
        self.M = 2
        self.n = 100
        self.res = 1000
        self.X = []

    def calc_R(self):
        data = self.data.ix[0:self.n, :]
        self.X = signal.hilbert(data.T)
        self.X = self.X.T
        R = [[0 for i in range(self.M)] for j in range(self.M)]

        for i in range(self.M):
            for j in range(self.M):
                R_ij = 0
                for k in range(self.n):
                    R_ij += self.X[k, i] * np.comjugate(self.X[k, j])
                R[i][j] = R_ij / self.n
        return R

    def calc_spec(self, R):
        W, V = LA.eig(R)
        E = V[:, 5:]
        div = np.pi / self.res
        start = - np.pi / 2
        music_spec = np.empty(self.res)

        for x in range(self.res):
            sin = math.sin(start)
            a = np.empty(10, dtype=complex)
            for i in range(10):
                s = pow(np.e, (complex(0, -(i+1)*np.pi)*sin))
                a[i] = s
            a_el = a.reshape((1, 10))
            a_el = np.conjugate(a_el)
            E_el = np.conjugate(E.T)
            numerator = np.dot(a_el, a)
            b_1 = np.dot(a_el, E)
            b_2 = np.dot(b_1, E_el)
            denominator = np.dot(b_2, a)
            music_spec[x] = 10 * (np.log10(numerator.real/denominator.real))
            start += div
        return music_spec

    def plot_x(self, music_spec):
        x_axis = np.linspace(-90, 90, self.res):
        plt.plot(x_axis, music_spec)
        plt.xlabel("Angle[angle]")
        plt.ylabel("Amplitude[db]")
        plt.show()

if __name__ == "__main__":
    music = Music()
    R = music.calc_R()
    spec = music.calc_spec(R)
    music.plot_x(spec)
