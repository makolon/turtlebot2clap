import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy.linalg as LA
import math
# read a path of the file of .csv
data = pd.read_csv("to.csvPath")
M = 2 #センサアレイの数
n = 100 #スナップショット数,つまり何個のデータセットを使うかということ
res = 1000 #推定する周波数分解能 180/res°になる
data = data.ix[0:n, :]
X =[] #センサデータを格納する行列(n×Mになるようにする)
X = signal.hilbert(data.T) #Xはデータそのままではなくヒルベルト変換を行う
X = X.T 
R = [[0 for i in range(M)] for j in range(M)] #R is the Covariance matrix of X

for i in range(M):
　　for j in range(M):
　　　　R_ij = 0
       for k in range(n):
　　       R_ij += X[k,i]* np.conjugate(X[k,j])
　　   R[i][j] = R_ij/n
W, V = LA.eig(R) #Rの固有値W,固有ベクトルV
E = V[:,5:] #固有値の小さい固有ベクトルから順番に音源の数だけ抜きとりEとする
e = np.e #Naipier's number
div = np.pi/res #Resolution

start = -np.pi/2
music_spec = np.empty(res) #MUSIC spectle. One peak for a index of coming direction.

for x in range(res):
    sin = math.sin(start)
    a = np.empty(10, dtype= complex)
    for i in range(10):
        s = pow(e, (complex(0, -(i+1)*np.pi)*sin))
        a[i] = s
    a_el = a.reshape((1,10))
    a_el = np.conjugate(a_el)
    E_el = np.conjugate(E.T)
    bunsi = np.dot(a_el, a)
    b_1 = np.dot(a_el, E)
    b_2 = np.dot(b_1, E_el)
    bunbo = np.dot(b_2, a)
    music_spec[x] = 10*(np.log10(bunsi.real/bunbo.real))
    start += div

x_axis = np.linspace(-90,90,res)
plt.plot(x_axis, music_spec)
plt.xlabel("Angle[°]")
plt.ylabel("Amplitude[dB]")
plt.show()
