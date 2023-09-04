from Gaussian_model import SSGaussianMixture
from Classify_model import BaseClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import numpy as np

if __name__ == '__main__':
    su = 500
    # kf = KFold(n_splits=5, shuffle=False)
    pca = PCA(n_components=50)
    # mnist = fetch_openml('mnist_784', cache=True)[0:2000]
    X = np.load(r'./dataset/MNIST.npy', allow_pickle=True)[0:4005]
    Y = np.load(r'./dataset/MNIST_label.npy', allow_pickle=True)[0:4005].astype(int)
    # gmm = SSGaussianMixture(748,10)
    cl = BaseClassifier()
    X = pca.fit_transform(X)
    # gmm.fit(X,)
    count = 0
    # X_train = X[0:su]
    # Y_train = Y[0:su]
    # X_test = X[100:9100]
    cl.fit(X, Y, su)
    # for train_index, test_index in kf.split(X):
    #     score = 0
    #     count += 1
    #     X_train, X_test = X[train_index], X[test_index]
    #     Y_train, Y_test = Y[train_index], Y[test_index]
    #     # gmm.fit(X_train,Y_train,Y_test)
    #     cl.fit(X_train, Y_train, X_test)
