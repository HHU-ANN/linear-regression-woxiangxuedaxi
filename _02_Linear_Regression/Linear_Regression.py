# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

# from sklearn import preprocessing


try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    x, y = read_data()
    t = 0.08
    weight = np.matmul(np.linalg.inv(np.matmul(x.T, x) + t * np.eye(6)), np.matmul(x.T, y))
    return weight @ data


def lasso(data):
    a = 0.07
    t = 0.05
    x, y = read_data()
    wei = np.array([1, 1, 1, 1, 1, 1])
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x = min_max_scaler.fit_transform(x)
    # y = min_max_scaler.fit_transform(y)
    wei = a * np.matmul(np.linalg.inv(np.matmul(x.T, x)), np.matmul(x.T, y) - t*wei)
    return wei @ data




def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
