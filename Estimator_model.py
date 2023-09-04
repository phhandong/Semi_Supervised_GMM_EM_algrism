from Classify_model import BaseClassifier
import numpy as np

class ConsolEstimator(object):
    def __init__(self, ids):
        self.mus = 0
        self.sigmas = 0
        self.pis = 0
        self.score_ini = 0
        self.clfs = {}
        self.id_column = 'wheezy-copper-turtle-magic'
        self.ids = ids

    def predict(self, df_X):
        y_pred = np.zeros(shape=(len(df_X)))
        for id in df_X[self.id_column].unique():
            id_rows = (df_X[self.id_column] == id)
            X = df_X.drop(['id', self.id_column], axis=1).values[id_rows]
            y_pred[id_rows] = self.clfs[id].predict(X)
        return y_pred

    def fit(self, df_train, df_test):
        for i, id in enumerate(self.ids):
            print(i, 'th training...') #df_train 有标签训练, df_test 无标签测试
            df_train_id = df_train[df_train[self.id_column] == id] # 加载数据 目标栏=id
            df_test_id = df_test[df_test[self.id_column] == id]
            if len(df_train_id) == 0 or len(df_test_id) == 0:
                continue
            else:
                print(id, 'th complete')

            X_train = df_train_id.drop(['id', 'target', self.id_column], axis=1).values #无训练标签数据
            y_train = df_train_id['target'].values #X_train 对应的标签
            X_test = df_test_id.drop(['id', self.id_column], axis=1).values #无测试标签数据

            self.clfs[id] = BaseClassifier()
            self.clfs[id].fit(X_train, y_train, X_test)

        print('mean score = ', np.array([clf.score for clf in self.clfs.values()]).mean())
        for index in self.clfs:
            if self.score_ini < self.clfs[index].score:
                self.score_ini = self.clfs[index].score
                self.mus = self.clfs[index].cgm.mus
                self.sigmas = self.clfs[index].cgm.sigmas
                self.pis = self.clfs[index].cgm.pis
                np.save(r'./params/mus.npy', self.mus)
                np.save(r'./params/sigmas.npy', self.sigmas)
                np.save(r'./params/pis.npy', self.pis)


        # self.b_mus = self.clfs.mus
        # self.b_sigmas = self.clfs.sigmas
        # self.b_pis = self.clfs.pis
