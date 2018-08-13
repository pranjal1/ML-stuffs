import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore",category=FutureWarning) 


X_train = np.arange(100).reshape(100,1)

Y_train = 4 * X_train + 1.5 + np.random.rand(100,1)*50


X_b = np.c_[np.ones((100, 1)), X_train]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y_train)

print theta_best

Y_predict = X_b.dot(theta_best)
plt.plot(X_train, Y_predict, "r-")
plt.plot(X_train, Y_train, "b.")
plt.axis([0, 100, 0, 1000])
plt.show()


