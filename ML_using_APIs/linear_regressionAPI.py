import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore",category=FutureWarning) 


X_train = np.arange(100).reshape(100,1)

Y_train = 4 * X_train + 1.5 + np.random.rand(100,1)*50


lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)


print lin_reg.intercept_, lin_reg.coef_

Y_predict = lin_reg.predict(X_train)


plt.plot(X_train, Y_predict, "r-")
plt.plot(X_train, Y_train, "b.")
plt.axis([0, 100, 0, 1000])
plt.show()


