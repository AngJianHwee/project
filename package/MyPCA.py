'''
# -*- coding: utf-8 -*-
# @Author: Ang Jian Hwee <angjianhwee@gmail.com>
# @Date:   2023-03-01 14:19:40
# @Last Modified by:   Ang Jian Hwee <angjianhwee@gmail.com>
# @Last Modified time: 2023-03-01 16:28:21
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class MyPCA:
    def __init__(self, n_components, df):
        self.pca = PCA(n_components=n_components)
        self.df = df
        self.X = df.values
        self.col_names_X = df.columns

    def fit_transform(self):
        self.X_pca = self.pca.fit_transform(self.X)

        self.pcs_composition = []
        for i in range(len(self.pca.components_)):
            self.pcs_composition.append({})
            for j in range(len(self.pca.components_[i])):
                self.pcs_composition[i][self.col_names_X[j]
                                        ] = self.pca.components_[i][j].round(4)

        self.explain_variance = self.pca.explained_variance_ratio_.round(4)
        self.cumulative_explain_variance = np.cumsum(
            self.explain_variance).round(4)

    def report(self):
        print('Principal Components Composition:')
        for i in range(len(self.pcs_composition)):
            print('PC{} = '.format(i+1), end='')
            print(
                ' + '.join([f"{self.pcs_composition[i][k]:>8.4f}*{k}" for k in self.pcs_composition[i]]), end="")
            print()
        print()

        print('Explained Variance Ratio:')
        for i in range(len(self.explain_variance)):
            print('PC{} = {}'.format(i+1, self.explain_variance[i]))
        print()

        print('Cumulative Explained Variance Ratio:')
        for i in range(len(self.cumulative_explain_variance)):
            print('PC{} = {}'.format(i+1, self.cumulative_explain_variance[i]))
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

    # Demo 1 - PCA with 2 components
    print('------------------ Demo 1 - PCA with 2 components ------------------')
    my_pca_instance = MyPCA(2, data)    # create an instance of MyPCA
    my_pca_instance.fit_transform()    # fit and transform the data
    my_pca_instance.report()    # print the report

    # Demo 2 - PCA with 3 components
    print('------------------ Demo 2 - PCA with 3 components ------------------')
    my_pca_instance = MyPCA(3, data)    # create an instance of MyPCA
    my_pca_instance.fit_transform()    # fit and transform the data
    my_pca_instance.report()    # print the report
