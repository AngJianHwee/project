# one liner PCA and EFA

## data
```
------------------ Sample Data (with shuffling) ------------------
   sepal_length  sepal_width  petal_length  petal_width  class
0           6.7          3.1           4.7          1.5      1
1           6.3          3.3           4.7          1.6      1
2           4.8          3.0           1.4          0.1      0
3           4.8          3.0           1.4          0.3      0
4           5.8          2.7           5.1          1.9      2
```

## PCA
- Code:
    ```
    # Demo 1 - PCA with 2 components
    print('------------------ Demo 1 - PCA with 2 components ------------------')
    my_pca_instance = MyPCA(2, data)    # create an instance of MyPCA
    my_pca_instance.fit_transform()    # fit and transform the data
    my_pca_instance.report()    # print the report
    ```
- Output:
    ```
    ------------------ Demo 1 - PCA with 2 components ------------------
    Principal Components Composition:
    PC1 =   0.3342*sepal_length +  -0.0783*sepal_width +   0.8005*petal_length +   0.3371*petal_width +   0.3575*class
    PC2 =   0.6886*sepal_length +   0.6841*sepal_width +  -0.0988*petal_length +  -0.0682*petal_width +  -0.2085*class

    Explained Variance Ratio:
    PC1 = 0.9226
    PC2 = 0.0481

    Cumulative Explained Variance Ratio:
    PC1 = 0.9226
    PC2 = 0.9707
    ```

## EFA
- Code:
    ```
    # Demo 1 - EFA with 2 components
    print('------------------ Demo 1 - EFA with 2 components ------------------')
    efa = MyEFA(n_components=2, df=data.iloc[:, :-1])
    efa.fit_transform()
    efa.report()
    ```
- Output:
    ```
    ------------------ Demo 1 - EFA with 2 components ------------------
    Factors Composition:
    Factor1 =   0.7258*sepal_length +  -0.1775*sepal_width +   1.7573*petal_length +   0.7320*petal_width
    Factor2 =  -0.3704*sepal_length +  -0.2406*sepal_width +   0.0279*petal_length +   0.0412*petal_width

    Explained Variance Ratio:
    Factor1 = 0.0174
    Factor2 = 0.0973
    Factor3 = 0.0034
    Factor4 = 0.0411
    ```
