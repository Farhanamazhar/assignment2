# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:24:05 2020

@author: FARHANA MAZHAR
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
g=dataset.loc[dataset.State=='New York',:]
gt=dataset.loc[dataset.State=='California' ,:]

yn=g.iloc[:,-1].values
w=np.arange(17)
xn=w.reshape(-1,1)
yc=gt.iloc[:,-1].values
f=np.arange(17)

xc=f.reshape(-1,1)



# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xn, yn, test_size = 0.2, random_state = 0)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(xn, yn)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 8)
X_poly = poly_reg.fit_transform(xn)
poly_reg.fit(X_poly, yn)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, yn)

# Visualising the Linear Regression results
plt.scatter(xn, yn, color = 'red')
plt.plot(xn, lin_reg.predict(xn), color = 'blue')
plt.title('NEWYORK')
plt.xlabel('STARTUPS')
plt.ylabel('PROFIT')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(xn, yn, color = 'red')
plt.plot(xn, lin_reg_2.predict(poly_reg.fit_transform(xn)), color = 'blue')
plt.title('NEWYORK')
plt.xlabel('STATUPS')
plt.ylabel('PROFITS')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(xn), max(xn), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(xn, yn, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('NEWYORK')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# Splitting the datasetinto the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xc, yc, test_size = 0.2, random_state = 0)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regc = LinearRegression()
lin_regc.fit(xc, yc)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regc = PolynomialFeatures(degree = 8)
X_polyc = poly_regc.fit_transform(xc)
poly_regc.fit(X_polyc, yc)
lin_reg_2c = LinearRegression()
lin_reg_2c.fit(X_polyc, yc)

# Visualising the Linear Regression results
plt.scatter(xc, yc, color = 'orange')
plt.plot(xc, lin_regc.predict(xc), color = 'yellow')
plt.title('CALIFORNIA')
plt.xlabel('startups')
plt.ylabel('profit')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(xc, yc, color = 'orange')
plt.plot(xc, lin_reg_2c.predict(poly_regc.fit_transform(xn)), color = 'yellow')
plt.title('CALIFORNIA')
plt.xlabel('startup')
plt.ylabel('profit')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
Xc_grid = np.arange(min(xc), max(xc), 0.1)
Xc_grid = Xc_grid.reshape((len(Xc_grid), 1))
plt.scatter(xc, yc, color = 'red')
plt.plot(Xc_grid, lin_reg_2c.predict(poly_regc.fit_transform(Xc_grid)), color = 'blue')
plt.title('CALIFORNIA')
plt.xlabel('startup')
plt.ylabel('profit')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])
NEWYORK=lin_reg_2c.predict(poly_regc.fit_transform([[6.5]]))
# Predicting a new result with Polynomial Regression
print('NEWYORK=',lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
print('CALIFORNIA=',lin_reg_2c.predict(poly_regc.fit_transform([[6.5]])))
A=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
B=lin_reg_2c.predict(poly_regc.fit_transform([[6.5]]))

if (A < B):
   st='CALIFORNIA WILL GET MORE PROFIT'
else:
  st="NEWYORK WILL GET MORE PROFIT" 
print (st)
