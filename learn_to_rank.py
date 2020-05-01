__author__ = '{Alfonso Aguado Bustillo}'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as linear_model
import lightgbm as lgb


class Partitions:

    def __init__(self, x_train=None, x_val=None, x_test=None, y_train=None, y_val=None, y_test=None):

        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def create_data_partitions(self, x, y):

        # split in train and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        # split train and validation data
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train,
                                                                              test_size=0.25, random_state=0)
        # feature Scaling
        sc = StandardScaler()
        self.x_train = sc.fit_transform(self.x_train)
        self.x_val = sc.fit_transform(self.x_val)
        self.x_test = sc.transform(self.x_test)


class LinearRegression:

    def __init__(self, partitions=None, coeff=None, x_train=None, x_test=None, y_train=None, y_test=None, y_pred=None):
        self.partitions = partitions  # train and test data partitions
        self.coeff = coeff
        self.y_pred = y_pred

    def execute(self):

        # create model
        model = linear_model.LinearRegression()

        # train
        model.fit(self.partitions.x_train, self.partitions.y_train)

        # get regression coefficients
        self.coeff = model.coef_

        # prediction
        y_pred = model.predict(self.partitions.x_test)
        self.y_pred = y_pred

        return y_pred, self.partitions.y_test


class LightGBM:

    def __init__(self, partitions, hyperparams, y_pred=None):

        self.hyperparams = hyperparams  # hyperparameters
        self.partitions = partitions  # train and test data partitions
        self.y_pred = y_pred

    def execute(self):

        # convert dataset into LightGMB format
        d_train = lgb.Dataset(self.partitions.x_train, label=self.partitions.y_train)

        # train
        clf = lgb.train(self.hyperparams, d_train, 100)

        # prediction
        y_pred = clf.predict(self.partitions.x_test)

        return y_pred, self.partitions.y_test


class LambdaRank:

    def __init__(self, partitions, y_pred=None):

        self.partitions = partitions  # train and test data partitions
        self.y_pred = y_pred

    def execute(self):

        gbm = lgb.LGBMRanker()

        q_train = [self.partitions.x_train.shape[0]]
        q_val = [self.partitions.x_val.shape[0]]

        # train
        gbm.fit(self.partitions.x_train, self.partitions.y_train, group=q_train,
                eval_set=[(self.partitions.x_val, self.partitions.y_val)],
                eval_group=[q_val], early_stopping_rounds=20, verbose=False,
                callbacks=[lgb.reset_parameter(learning_rate=lambda x: 0.95 ** x * 0.1)])

        y_pred = gbm.predict(self.partitions.x_test)

        return y_pred, self.partitions.y_test
