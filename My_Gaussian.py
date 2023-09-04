from Gaussian_model import SSGaussianMixture
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ['OMP_NUM_THREADS'] = '4'

if __name__ == '__main__':
    # gau_model = SSGaussianMixture(50, 10)
    # 创建高斯混合模型对象
    # gmm = GaussianMixture(n_components=10, covariance_type='diag')
    gmm = GaussianMixture(n_components=10)
    # 训练模型
    # X_train = [[0], [1], [2], [3], [4], [5]]
    # gmm.fit(X_train)
    su = 2000
    kf = KFold(n_splits=5, shuffle=False)
    # mnist = fetch_openml('mnist_784', cache=True)[0:2000]
    X = np.load(r'./dataset/MNIST.npy', allow_pickle=True)[0:5000]
    Y = np.load(r'./dataset/MNIST_label.npy', allow_pickle=True)[0:5000]
    count = 0
    for train_index, test_index in kf.split(X):
        score = 0
        count += 1
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        gmm.fit(X_train[0:su], Y_train[0:su].reshape(1, -1))
        gmm.fit(X_train[su + 1:])
        output = gmm.predict(X_test)
        for i, j in zip(output, Y_test):
            if i == int(j):
                score += 1
        # for X_item, Y_item in zip(X_train[0:su], Y_train[0:su]):
        #     gmm.fit(X_item.reshape(-1, 1), Y_item)
        # for X_item in X_train[su + 1:]:
        #     gmm.fit(X_item.reshape(-1, 1))
        # for test_item, label in zip(X_test, Y_test):
        #     output = gmm.predict(test_item.reshape(-1, 1))
        #     score += (label == output)
        print("The Group {:} accuracy is: {:.6f} ".format(count, score/1000))

    # 预测新样本
    # X_test = [[1.5], [3.5], [6]]
    # y_pred = gmm.predict(X_test)
    #
    # print(y_pred)
