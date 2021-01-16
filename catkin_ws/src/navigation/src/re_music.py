import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy.linalg as LA
import math

data = pd.read_csv("/home/makolon/turtlebot2clap/catkin_ws/src/navigation/src/output_Output_stereo.csv")

M = 2
n = len(data)
res = 44100
data = data.iloc[0:n, :]
X = []
X = signal.hilbert(data.T)
X = X.T
R = [[0 for i in range(M)] for j in range(M)]
for i in range(M):
    for j in range(M):
	R_ij = 0
	for k in range(n):
	    R_ij += X[k, i] + np.conjugate(X[k, j])
        R[i][j] = R_ij / n
W, V = LA.eig(R)
E = V[:, 1:]
e = np.e
div = np.pi / res

start = - np.pi /2
music_spec = np.empty(res)

for x in range(res):
    sin = math.sin(start)
    a = np.empty(M, dtype=complex)
    for i in range(M):
	s = pow(e, (complex(0, -(i+1)*np.pi)*sin))
	a[i] = s
    a_el = a.reshape((1, M))
    a_el = np.conjugate(a_el)
    E_el = np.conjugate(E.T)
    bunsi = np.dot(a_el, a)
    b_1 = np.dot(a_el, E)
    b_2 = np.dot(b_1, E_el)
    bunbo = np.dot(b_2, a)
    music_spec[x] = M * (np.log10(bunsi.real / bunbo.real))
    start += div

x_axis = np.linspace(-90, 90, res)
plt.plot(x_axis, music_spec)
plt.xlabel("Angle")
plt.ylabel("Amplitude")
plt.show()
    
