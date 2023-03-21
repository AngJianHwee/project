# Import libraries
import package
import pprint
import package.MyPCA as MyPCA
import pandas as pd  # Library for data manipulation
import matplotlib.pyplot as plt
import numpy as np

# fix matplotlib font issue for chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# Read excel file
df = pd.read_excel('11918392_202303191156582830.xlsx')

# save the columns to a file called df_columns_hard_coded.txt
# df_columns_hard_coded = df.columns
# with open('df_columns_hard_coded.txt', 'w', encoding='utf-8') as f:
#     for item in df_columns_hard_coded:
#         f.write(item + "\n")

# read from the file
# with open('df_columns_hard_coded.txt', 'r', encoding='utf-8') as f:
#     df_columns_hard_coded = f.read().splitlines()

# exclude these
# 1.性别
# 2.年龄
# 3.地区
# 3.地区[选项填空]
# 4.您目前所从事的行业
# 4.您目前所从事的行业[选项填空]
# 5.您来澳门旅游的目的:观光旅游
# 5.您来澳门旅游的目的:赌博娱乐
# 5.您来澳门旅游的目的:购物消费
# 5.您来澳门旅游的目的:美食文化
# 5.您来澳门旅游的目的:商务考察
# 5.您来澳门旅游的目的:参观大学
# 5.您来澳门旅游的目的:其他____{fillblank-345b}
# 5.您来澳门旅游的目的:其他____{fillblank-345b}[选项填空]


# exclude first 14 columns and the last one
df = df.iloc[:, 14:-1]

# do PCA using package.MyPCA

# do PCA
pca = MyPCA(10, df)
pca.fit_transform()
# pca.report()

THRESHOLD = 0.1

for composition in pca.pcs_composition:
    # sort dict by absolute value inverse
    sorted_composition = (
        sorted(composition.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

    # filter those abs larger than threshold
    sorted_composition_sig = list(
        filter(lambda x: abs(x[1]) > THRESHOLD, sorted_composition))

    # pprint the result
    # pprint.pprint(sorted_composition_sig)

    # Add value to the label string, round to 2 decimal places
    labels = [x[0] + " " + str(round(x[1], 2)) for x in sorted_composition_sig]

    # for labels that are too long, split into 3 lines equally by adding "\n" at every len() % 3
    labels = [x[:len(x) // 3] + "\n" + x[len(x) // 3:2 * len(x) // 3] + "\n" + x[2 * len(x) // 3:] if len(x) > 10 else x for x in labels]

    # tight layout
    plt.tight_layout()

    # actual values are abs, but text on top is the original value
    values = [abs(x[1]) for x in sorted_composition_sig]

    # create pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)

    fig1.subplots_adjust(left=0.3, right=1-0.3)


    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')

    # add title dynamically with 2 leading zeros
    title = "PCA_" + str(pca.pcs_composition.index(composition) + 1).zfill(2)
    plt.title(title)

    # add a textbox
    plt.text(0.5, 0.5, str(labels), ha='center', va='center', size=10)

    
    # save the figure to pirChart/
    plt.savefig("./plots/pieChart/" + title + ".png")
