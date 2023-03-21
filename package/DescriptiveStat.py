'''
# -*- coding: utf-8 -*-
# @Author: Ang Jian Hwee <angjianhwee@gmail.com>
# @Date:   2023-03-10 11:39:34
# @Last Modified by:   Ang Jian Hwee <angjianhwee@gmail.com>
# @Last Modified time: 2023-03-10 17:42:49
'''


class DescriptiveStat:
    def __init__(self, df):
        self.df = df
        self.descriptive_stat = self._generate_descriptive_stat()

    def _generate_descriptive_stat(self):
        # Generate descriptive statistics
        return self.df.describe()

    def boxplot(self, ignore_columns=[]):
        # Generate boxplot and plt.show, do not save
        self.df[[col for col in self.df.columns if col not in ignore_columns]].boxplot()
        plt.show()

    def histogram(self, ignore_columns=[]):
        # Generate histogram and plt.show, do not save
        self.df[[col for col in self.df.columns if col not in ignore_columns]].hist()
        plt.show()


if __name__ == '__main__':
    # Import libraries
    import pandas as pd  # Library for data manipulation
    import matplotlib.pyplot as plt  # Library for data visualization

    # sample iris data from built-in library
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',  # Load Iris dataset from UCI Machine Learning Repository
                     header=None)

    # rename the columns, using _ instead of space
    df.columns = ['sepal_length', 'sepal_width', 'petal_length',
                  'petal_width', 'class']  # Rename the columns of the dataset

    # encode the class without hard-coding convert to int using sklearn
    # Import LabelEncoder from scikit-learn
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()  # Create a LabelEncoder object
    # Encode the 'class' column using LabelEncoder
    df['class'] = le.fit_transform(df['class'])

    # Create object of DescriptiveStat class
    # Create an object of the DescriptiveStat class, which contains methods for generating descriptive statistics, boxplot, and histogram
    obj = DescriptiveStat(df)

    # Generate descriptive statistics and print
    # Call the '_generate_descriptive_stat' method of the 'obj' object and print the results to the console
    print(obj._generate_descriptive_stat())

    # Generate boxplot
    obj.boxplot()  # Call the 'boxplot' method of the 'obj' object to generate a boxplot of the dataset

    # Generate histogram
    obj.histogram()  # Call the 'histogram' method of the 'obj' object to generate a histogram of the dataset

    # Generate boxplot, ignore class column
    # Call the 'boxplot' method of the 'obj' object, but ignore the 'class' column when generating the boxplot
    obj.boxplot(ignore_columns=['class'])

    # Generate histogram, ignore class column
    # Call the 'histogram' method of the 'obj' object, but ignore the 'class' column when generating the histogram
    obj.histogram(ignore_columns=['class'])

    # Show plots
    plt.show()  # Display the plots generated by the 'boxplot' and 'histogram' methods using Matplotlib
