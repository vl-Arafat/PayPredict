import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Dataset load & model train
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# UI
st.title("💰 Salary Predictor")

level = st.number_input("Enter your position level (1-10):", min_value=1.0, max_value=10.0, step=0.5)

if st.button("Predict Salary"):
    pred = lin_reg_2.predict(poly_reg.fit_transform([[level]]))
    st.success(f"Predicted Salary: ${pred[0]:,.2f}")