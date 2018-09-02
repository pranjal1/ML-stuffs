import matplotlib.pyplot as plt
import numpy as np
import math


x1 = np.arange(1000,2000).reshape(1000,-1) * np.pi/180
x2 = np.zeros(x1.shape)
x3 = np.zeros(x1.shape)
for i in range(len(x1)):
	x2[i] = math.sin(x1[i])	 #perfect value
	x3[i] = x2[i] + np.random.randn()/3 #scattered values

alpha = 0.95
v = 0
v0 = np.zeros(x2.shape)
v0_bias_corrected = np.zeros(x2.shape)
for i in range(len(x2)):
	v0[i] = alpha * v + (1-alpha) * x2[i] #exponentially weighted averages
	v0_bias_corrected[i] = v0[i] / (1-alpha**i)#bias correction
	v = v0[i]


fig = plt.figure()
plt.scatter(x1,x2,s=2,c='Red',cmap=plt.cm.Spectral)
plt.scatter(x1,x3,s=2,c='Green',cmap=plt.cm.Spectral)
plt.scatter(x1,v0,s=2,c='Pink',cmap=plt.cm.Spectral)
plt.scatter(x1,v0_bias_corrected,s=2,cmap=plt.cm.Spectral)
plt.show()
#look at the starting point of pink line and blue line


