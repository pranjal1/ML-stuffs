import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore",category=FutureWarning) 


m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)


X_poly = poly_features.fit_transform(X) #np.c_[X_poly,X,X**2]


#X_poly now has X and X^2, we use linear regression to fit this data
#so, we get fitting for theta0+theta1*X+theta2*X^2

lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)


print lin_reg.intercept_, lin_reg.coef_

y_predict = lin_reg.predict(X_poly)


plt.plot(X, y_predict, "r.")
plt.plot(X, y, "b.")
plt.axis([-5, 5, 0, 10])
plt.show()


