from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle

inputx = open('triple_pendulum_angle_one_lots.pkl','rb')
inputy = open('triple_pendulum_angle_two_lots.pkl','rb')
inputz = open('triple_pendulum_angle_three_lots.pkl','rb')

X = pickle.load(inputx)
Y = pickle.load(inputy)
Z = pickle.load(inputz)

inputx.close()
inputy.close()
x = []
y = []
z = []

for i in range(len(X)):
    if(X[i] < 1.58 and X[i] > -1.58):
        x.append(X[i])
        y.append(Y[i])
        z.append(Z[i])

fig = plt.figure()
ax = fig.gca(projection = '3d')
c = x
ax.scatter(x,y,z, c = c)
ax.set_xlabel('angle_1')
ax.set_ylabel('angle_2')
ax.set_zlabel('angle_3')
fig.suptitle('Triple Pendulum Equilibrium Points')
plt.grid(True)
plt.show()

X = []
Y = []
Z = []

for i in range(len(x)):
#    if(y[i] < 3.141 and y[i] > 3.12):
#        X.append(x[i])
#        Y.append(y[i])
#        Z.append(z[i])
    if(z[i] < 3.141 and z[i] > 3.13):
        X.append(x[i])
        Y.append(y[i])
        Z.append(z[i])
X2 = []
Y2 = []
Z2 = []

for i in range(len(x)):
    if(y[i] < 3.141 and y[i] > 3.13):
        X2.append(x[i])
        Y2.append(y[i])
        Z2.append(z[i])
X3 = []
Y3 = []
Z3 =[]
for i in range(len(x)):
    if(y[i] < -3.13 and y[i] > -3.14):
        X3.append(x[i])
        Y3.append(y[i])
        Z3.append(z[i])

X4=[]
Y4=[]
Z4=[]
for i in range(len(x)):
    if(z[i] > -3.141 and z[i] < -3.13):
        X4.append(x[i])
        Y4.append(y[i])
        Z4.append(z[i])

threshold = 0.1
foundz = []
foundx = []
foundy = []
for i in range(len(X)):
    for j in range(len(x)):
        if(x[j] < (X[i]+threshold) and x[j] > (X[i] - threshold) and y[j] < (Y[i] + threshold) and y[j] > (Y[i] - threshold)):
            foundx.append(x[j])
            foundy.append(y[j])
            foundz.append(z[j])

for i in range(len(X2)):
    for j in range(len(x)):
        if(x[j] < (X2[i]+threshold) and x[j] > (X2[i] - threshold) and z[j] < (Z2[i] + threshold) and z[j] > (Z2[i] - threshold)):
            foundx.append(x[j])
            foundy.append(y[j])
            foundz.append(z[j])
for i in range(len(X3)):
    for j in range(len(x)):
        if(x[j] < (X3[i]+threshold) and x[j] > (X3[i] - threshold) and y[j] < (Y3[i] + threshold) and y[j] > (Y3[i] - threshold)):
            foundx.append(x[j])
            foundy.append(y[j])
            foundz.append(z[j])
for i in range(len(X4)):
    for j in range(len(x)):
        if(x[j] < (X4[i]+threshold) and x[j] > (X4[i] - threshold) and z[j] < (Z4[i] + threshold) and z[j] > (Z4[i] - threshold)):
            foundx.append(x[j])
            foundy.append(y[j])
            foundz.append(z[j])



fig = plt.figure()
ax = fig.gca(projection = '3d')

c=foundx
ax.scatter(foundx, foundy, foundz, c=c)

ax.set_xlabel('angle_1')
ax.set_ylabel('angle_2')
ax.set_zlabel('angle_3')
fig.suptitle('Triple Pendulum Equilibrium Points')
plt.grid(True)
plt.show()



