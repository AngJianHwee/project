'''
# -*- coding: utf-8 -*-
# @Author: Ang Jian Hwee <angjianhwee@gmail.com>
# @Date:   2023-03-01 16:28:46
# @Last Modified by:   Ang Jian Hwee <angjianhwee@gmail.com>
# @Last Modified time: 2023-03-01 16:55:53
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import FactorAnalysis


class MyEFA:
    def __init__(self, n_components, df):
        self.efa = FactorAnalysis(n_components=n_components)
        self.df = df
        self.X = df.values
        self.col_names_X = df.columns

    def fit_transform(self):
        self.X_efa = self.efa.fit_transform(self.X)

        self.factors_composition = []
        for i in range(len(self.efa.components_)):
            self.factors_composition.append({})
            for j in range(len(self.efa.components_[i])):
                self.factors_composition[i][self.col_names_X[j]
                                            ] = self.efa.components_[i][j].round(4)

        self.explain_variance = self.efa.noise_variance_.round(4)

    def report(self):
        print('Factors Composition:')
        for i in range(len(self.factors_composition)):
            print('Factor{} = '.format(i+1), end='')
            print(
                ' + '.join([f"{self.factors_composition[i][k]:>8.4f}*{k}" for k in self.factors_composition[i]]), end="")
            print()
        print()

        print('Explained Variance Ratio:')
        for i in range(len(self.explain_variance)):
            print('Factor{} = {}'.format(i+1, self.explain_variance[i]))
        print()


if __name__ == "__main__":
    import pprint

    # sample iris data from built-in library
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                       header=None)

    # rename the columns, using _ instead of space
    data.columns = ['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width', 'class']

    # encode the class without hard-coding convert to int using sklearn
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data['class'] = le.fit_transform(data['class'])

    print('------------------ Sample Data (with shuffling) ------------------')
    print(data.sample(frac=1).reset_index(drop=True).head())

    # Demo 1 - EFA with 2 components
    print('------------------ Demo 1 - EFA with 2 components ------------------')
    efa = MyEFA(n_components=2, df=data.iloc[:, :-1])
    efa.fit_transform()
    efa.report()

    # Demo 2 - EFA with 3 components
    print('------------------ Demo 2 - EFA with 3 components ------------------')
    efa = MyEFA(n_components=3, df=data.iloc[:, :-1])
    efa.fit_transform()
    efa.report()
