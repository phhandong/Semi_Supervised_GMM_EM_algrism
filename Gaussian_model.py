import numpy as np
from scipy import stats


class SSGaussianMixture(object):
    def __init__(self, n_features, n_categories):
        super(SSGaussianMixture, self).__init__()
        self.n_features = n_features
        self.n_categories = n_categories

        self.mus = np.array([np.random.randn(n_features)] * n_categories)
        self.sigmas = np.array([np.eye(n_features)] * n_categories)
        self.pis = np.array([1 / n_categories] * n_categories)

    # def fit(self, X_train, y_train, X_test, threshold=0.00001, max_iter=100):
    def fit(self, X_train, y_train, su, threshold=0.00001, max_iter=100):
        Z_train = np.eye(self.n_categories)[y_train[0:su]]  # 给出su个有标签数据
        # X_train 为标签数据，y_train为标签，X_test为无标签数据
        for i in range(max_iter):
            # EM algorithm
            # M step
            Z_test = np.array([self.gamma(X_train[su + 1:], k) for k in range(self.n_categories)]).T  # 分类概率
            if np.any(Z_test.sum(axis=1, keepdims=True) == 0):
                Z_test /= Z_test.sum(axis=1, keepdims=True)

            # E step
            datas = [X_train[0:su], Z_train, X_train[su + 1:], Z_test]  # (20,50) (20,10) (9979,50) (9799,10)
            mus = np.array([self._est_mu(k, *datas) for k in range(self.n_categories)])
            sigmas = np.array([self._est_sigma(k, *datas) for k in range(self.n_categories)])
            pis = np.array([self._est_pi(k, *datas) for k in range(self.n_categories)])

            diff = max(np.max(np.abs(mus - self.mus)),
                       np.max(np.abs(sigmas - self.sigmas)),
                       np.max(np.abs(pis - self.pis)))  # 取最大值

            self.mus = mus
            self.pis = pis
            if not (np.any(np.isinf(sigmas)) | np.any(np.isnan(sigmas))):
                if np.all(np.linalg.eigvals(sigmas) > 0):
                    self.sigmas = sigmas
            else:
                self.sigmas = np.array([np.eye(self.n_features)] * self.n_categories)
                self.sigmas = np.array([np.eye(self.n_features)] * self.n_categories)
                self.pis = np.array([1 / self.n_categories] * self.n_categories)

            if diff < threshold:
                break

        print("The diff is: %s, The mus is: %s, The sigmas is: %s, The pis is: %s" % (
            diff, mus.mean(), sigmas.mean(), pis.mean()))

    def predict_proba(self, X):
        Z_pred = np.array([self.gamma(X, k) for k in range(self.n_categories)]).T
        Z_pred /= Z_pred.sum(axis=1, keepdims=True)
        return Z_pred

    def gamma(self, X, k):
        # X is input vectors, k is feature index
        # 计算多元正态分布的概率密度函数
        return stats.multivariate_normal.pdf(X, mean=self.mus[k], cov=self.sigmas[k])

    def _est_mu(self, k, X_train, Z_train, X_test, Z_test):
        mu = (Z_train[:, k] @ X_train + Z_test[:, k] @ X_test).T / \
             (Z_train[:, k].sum() + Z_test[:, k].sum())
        return mu

    def _est_sigma(self, k, X_train, Z_train, X_test, Z_test):
        cmp1 = (X_train - self.mus[k]).T @ np.diag(Z_train[:, k]) @ (X_train - self.mus[k])
        cmp2 = (X_test - self.mus[k]).T @ np.diag(Z_test[:, k]) @ (X_test - self.mus[k])
        sigma = (cmp1 + cmp2) / (Z_train[:, k].sum() + Z_test[:k].sum())
        return sigma

    def _est_pi(self, k, X_train, Z_train, X_test, Z_test):
        pi = (Z_train[:, k].sum() + Z_test[:, k].sum()) / \
             (Z_train.sum() + Z_test.sum())
        return pi
