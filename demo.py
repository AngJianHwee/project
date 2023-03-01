'''
# -*- coding: utf-8 -*-
# @Author: Ang Jian Hwee <angjianhwee@gmail.com>
# @Date:   2023-03-01 16:37:35
# @Last Modified by:   Ang Jian Hwee <angjianhwee@gmail.com>
# @Last Modified time: 2023-03-01 16:55:11
'''


import package
import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # sample data for regression from HTTP
    data = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')

    # rename the columns using "_"
    data.columns = data.columns.str.replace(' ', '_')

    # print the data
    print('------------------ Sample Data (with shuffling) ------------------')
    print(data.sample(frac=1).reset_index(drop=True).head())

    # split the data into training and testing
    df_X = data.drop('quality', axis=1)
    df_y = data['quality']
    X_train, X_test, y_train, y_test = train_test_split(
        df_X, df_y, test_size=0.2, random_state=42)

    # linear regression without PCA and EFA
    print('------------------ Linear Regression without PCA and EFA ------------------')
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print('MSE = {}'.format(mean_squared_error(y_test, y_pred)))
    print('R2 = {}'.format(lr.score(X_test, y_test)))

    # linear regression with PCA
    print('------------------ Linear Regression with PCA ------------------')
    pca = package.MyPCA(n_components=2, df=X_train)
    pca.fit_transform()
    pca.report()

    # transform the data
    X_train_pca = pca.X_pca
    X_test_pca = pca.pca.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_pca, y_train)
    y_pred = lr.predict(X_test_pca)
    print('MSE = {}'.format(mean_squared_error(y_test, y_pred)))
    print('R2 = {}'.format(lr.score(X_test_pca, y_test)))

    # linear regression with EFA
    print('------------------ Linear Regression with EFA ------------------')
    efa = package.MyEFA(n_components=2, df=X_train)
    efa.fit_transform()
    efa.report()

    # transform the data
    X_train_efa = efa.X_efa
    X_test_efa = efa.efa.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_efa, y_train)
    y_pred = lr.predict(X_test_efa)
    print('MSE = {}'.format(mean_squared_error(y_test, y_pred)))
    print('R2 = {}'.format(lr.score(X_test_efa, y_test)))
