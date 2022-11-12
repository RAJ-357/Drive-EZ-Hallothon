import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, accuracy_score

def polreg(x,input):
    dataset = pd.read_csv(x)
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    poly_reg = PolynomialFeatures(degree = 4)
    X_poly = poly_reg.fit_transform(X)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)

    return lin_reg_2.predict(poly_reg.fit_transform([[input]]))[0]

#polreg('Position_Salaries.csv')
