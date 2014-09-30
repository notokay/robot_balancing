from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend, rcParams
import matplotlib.animation as animation
import pickle

inputx = open('double_pen_angle_1.pkl','rb')
inputy = open('double_pen_angle_2.pkl','rb')

X = pickle.load(inputx)
Y = pickle.load(inputy)

inputx.close()
inputy.close()

plt.scatter(X, Y)
xlabel('angle_1')
ylabel('angle_2')
plt.grid(True)
plt.show()

