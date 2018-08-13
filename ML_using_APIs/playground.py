import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

X = np.arange(30).reshape(10,3)
X[:,2] =0
print X



fig = plt.figure()
Axes3D = fig.add_subplot(111, projection='3d')
Axes3D.scatter(X[:,0],X[:,1],X[:,2],s=10)
plt.show()
