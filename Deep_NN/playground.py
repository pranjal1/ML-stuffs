import numpy as np
import matplotlib.pyplot as plt

X1 = np.random.rand(1000,1)*1000
X2 = np.random.rand(1000,1)*10


X = np.c_[X1,X2]

X1mean = X1 - np.mean(X[:,0])
X2mean = X2 - np.mean(X[:,1])

XMeanReduced = np.c_[X1mean,X2mean]

X1std = X1mean/np.std(X[:,0])
X2std = X2mean/np.std(X[:,1])


XstdDiv = np.c_[X1std,X2std]

print min(X1),max(X1),min(X2),max(X2)
print min(X1mean),max(X1mean),min(X2mean),max(X2mean)
print min(X1std),max(X1std),min(X2std),max(X2std)


fig = plt.figure()
plt.scatter(X[:,0], X[:,1],c='Red',s=20, cmap=plt.cm.Spectral)
plt.scatter(XMeanReduced[:,0], XMeanReduced[:,1],c='Blue',s=20, cmap=plt.cm.Spectral)
#plt.scatter(XstdDiv[:,0], XstdDiv[:,1],c='Green',s=20, cmap=plt.cm.Spectral)
#plt.axis([-100, 100, -100, 100])
plt.show()

