import sklearn.linear_model as linear_model
from lightgbm import LGBMRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import lightgbm as lgb
import random
import numpy as np
from sklearn.preprocessing import StandardScaler


class Partitions:

    def __init__(self, x_train=None, x_val=None, x_test=None, y_train=None, y_val=None, y_test=None):

        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def create_data_partitions(self, x, y, normalized=False, with_validation=False):

        # split in train and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        if normalized:
            # feature Scaling
            sc_x = StandardScaler(with_mean=False)
            sc_y = StandardScaler(with_mean=False)
            self.x_train = sc_x.fit_transform(self.x_train)
            self.y_train = sc_y.fit_transform(self.y_train)
            self.x_test = sc_x.fit_transform(self.x_test)
            self.y_test = sc_y.fit_transform(self.y_test)

        if with_validation:
            # split train and validation data
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train,
                                                                                  test_size=0.25, random_state=42)
            # feature Scaling
            self.y_val = np.log(self.y_val)
            if normalized:
                # feature Scaling
                self.x_val = sc_x.fit_transform(self.x_val)
                self.y_val = sc_y.fit_transform(self.y_val)


def get_minibatch(x, y, indexes, size):
    if len(indexes) < size:
        sample_indexes = indexes
        indexes = []
    else:
        sample_indexes = random.sample(list(indexes), size)
        indexes = np.delete(indexes, np.in1d(indexes, sample_indexes))

    return x[sample_indexes], y[sample_indexes], indexes


class DummyRegressor:

    def __init__(self, partitions):
        self.partitions = partitions  # train and test data partitions
        self.y_pred = None
        self.y_score = None

    def execute(self):
        from sklearn.dummy import DummyRegressor

        dummy_regr = DummyRegressor(strategy="mean")
        dummy_regr.fit(self.partitions.x_train, self.partitions.y_train)
        y_pred = dummy_regr.predict(self.partitions.x_test)
        self.y_pred = y_pred

        return self.y_pred, self.partitions.y_test


class LinearRegression:

    def __init__(self, partitions):
        self.partitions = partitions  # train and test data partitions
        self.coeff = None
        self.y_pred = None

    def execute(self):

        # create model
        # model = linear_model.LinearRegression(fit_intercept=True, n_jobs=-1)
        model = linear_model.LinearRegression(n_jobs=-1)

        # train (no batches)
        model.fit(self.partitions.x_train, self.partitions.y_train)

        # get regression coefficients
        self.coeff = model.coef_

        # prediction
        y_pred = model.predict(self.partitions.x_test)
        self.y_pred = y_pred

        return y_pred, self.partitions.y_test


class LinearRegressionBatch:

    def __init__(self, partitions):
        self.partitions = partitions  # train and test data partitions
        self.coeff = None
        self.y_pred = None
        self.score = None

    def execute(self):

        # create model
        model = linear_model.SGDRegressor(n_jobs=-1)

        # indexes to be used for getting minibatches
        indexes = np.arange(self.partitions.x_train.shape[0])

        while len(indexes) > 0:
            # get minibatches
            x_train, y_train, indexes = get_minibatch(self.partitions.x_train, self.partitions.y_train, indexes,
                                                      size=10000)

            model.partial_fit(x_train, y_train)

        # get regression coefficients
        self.coeff = model.coef_

        # prediction
        y_pred = model.predict(self.partitions.x_test)
        self.y_pred = y_pred

        return y_pred, self.partitions.y_test


class LinearRegressionRFECV:

    def __init__(self, partitions):
        self.partitions = partitions  # train and test data partitions
        self.n_features = None
        self.best_features = None
        self.feature_ranking = None

    def execute(self):

        # create model
        model = linear_model.SGDRegressor()

        # recursively eliminate features
        rfecv = RFECV(estimator=model, step=1, scoring="neg_mean_squared_error")
        rfecv.fit(self.partitions.x_train, self.partitions.y_train)
        rfecv.transform(self.partitions.x_train)

        # number of best features
        self.n_features = rfecv.n_features_

        # which categories are best
        self.best_features = rfecv.support_

        # rank features best (1) to worst
        self.feature_ranking = rfecv.ranking_

        return self.n_features, self.best_features, self.feature_ranking


class LightGBM:

    def __init__(self, partitions, best_params):

        # LightGBM training parameters
        # self.params = {'n_iter': 10000, 'boosting_type': 'gbdt', 'objective': 'regression', 'verbose': -1}

        self.params = {'n_iter': 10000, 'learning_rate': 0.001, 'boosting_type': 'gbdt', 'objective': 'regression',
                       'metric': 'mae', 'num_leaves': 350, 'max_depth': 9, 'min_data_in_leaf': 100, 'max_bin': 100,
                       'verbose': -1}

        if best_params:
            self.params = best_params

        self.partitions = partitions  # train and test data partitions
        self.y_pred = None

    def execute(self):

        # create model
        estimator = LGBMRegressor(boosting_type='gbdt', objective='regression')

        estimator = lgb.train(self.params,
                              init_model=estimator,
                              train_set=lgb.Dataset(self.partitions.x_train.toarray(),
                                                    self.partitions.y_train.flatten()))

        # prediction
        y_pred = estimator.predict(self.partitions.x_test)
        self.y_pred = y_pred

        return y_pred, self.partitions.y_test


class LightGBMBatch:

    def __init__(self, partitions):

        # LightGBM training parameters

        # For 25000
        self.params = {'n_iter': 10000, 'learning_rate': 0.001, 'boosting_type': 'gbdt', 'objective': 'regression',
                       'metric': 'mae', 'num_leaves': 350, 'max_depth': 9, 'min_data_in_leaf': 100,
                       'verbose': -1}

        # For 75000
        # self.params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metric': 'mae', 'learning_rate': 0.001,
        #                'max_bin': 255, 'max_depth': 12, 'min_data_in_leaf': 125, 'n_iter': 10000, 'num_leaves': 350,
        #                'verbose': -1}

        self.partitions = partitions  # train and test data partitions
        self.y_pred = None

    def execute(self):

        # size = 4460
        size = 15000
        estimator = None

        total_indexes = self.partitions.x_train.shape[0]
        index_count = 0
        while index_count < total_indexes:
            if (index_count + size) < total_indexes:
                indexes = list(range(index_count, index_count + size))
            else:
                indexes = list(range(index_count, total_indexes))

            index_count = index_count + len(indexes)

            estimator = lgb.train(self.params,
                                  init_model=estimator,
                                  train_set=lgb.Dataset(self.partitions.x_train[indexes].toarray(),
                                                        self.partitions.y_train[indexes].flatten()),
                                  keep_training_booster=True,
                                  num_boost_round=5000)

        # prediction
        y_pred = estimator.predict(self.partitions.x_test)
        self.y_pred = y_pred

        return y_pred, self.partitions.y_test


class LightGBMTuning:

    def __init__(self, partitions):

        self.partitions = partitions  # train and test data partitions
        self.y_pred = None
        self.best_params = None
        self.best_score = None

    def execute(self):

        # hyperparameter tuning
        param_grid = {'num_leaves': [350],
                      'max_depth': [9],
                      'learning_rate': [0.001, 0.002],
                      'min_data_in_leaf': [100],
                      'max_bin': [100],
                      # 'lambda_l1': [0.0, 0.01, 0.05, 1.0],
                      'lambda_l2': [0.0, 0.03, 1.0],
                      'n_iter': [10000]}

        # For 75000
        # {'learning_rate': 0.001, 'max_bin': 255, 'max_depth': 12, 'min_data_in_leaf': 125, 'n_iter': 10000,
        #  'num_leaves': 350}

        # create model
        estimator = LGBMRegressor(boosting_type='gbdt', objective='regression')

        search = GridSearchCV(estimator, param_grid, cv=7)
        search.fit(self.partitions.x_train, self.partitions.y_train)

        # prediction
        y_pred = search.predict(self.partitions.x_test)
        self.y_pred = y_pred
        self.best_score = search.best_score_
        self.best_params = search.best_params_

        return y_pred, self.partitions.y_test


class LightGBMRFECV:

    def __init__(self, partitions):
        self.partitions = partitions  # train and test data partitions
        self.n_features = None
        self.best_features = None
        self.feature_ranking = None

    def execute(self):

        # create model
        estimator = LGBMRegressor(boosting_type='gbdt', objective='regression', metric='mae',
                                  num_iterations=10000, learning_rate=0.001, num_leaves=350, max_depth=9,
                                  min_data_in_leaf=100)

        selector = RFECV(estimator, step=1, cv=4, scoring="neg_mean_squared_error", verbose=-1)
        selector.fit(self.partitions.x_train, self.partitions.y_train)

        # # prediction
        # y_pred = selector.predict(self.partitions.x_test)

        # select only the best features
        selector.fit(self.partitions.x_train[:, selector.support_], self.partitions.y_train)
        y_pred = selector.predict(self.partitions.x_test[:, selector.support_])

        # number of best features
        self.n_features = selector.n_features_

        # which categories are best
        self.best_features = selector.support_

        # rank features best (1) to worst
        self.feature_ranking = selector.ranking_

        return y_pred, self.partitions.y_test


class LogisticRegression:

    def __init__(self, partitions):
        self.partitions = partitions  # train and test data partitions
        self.coeff = None
        self.y_pred = None

    def execute(self):

        # create model
        # model = linear_model.LogisticRegression(max_iter=100000, C=0.1, class_weight='balanced', n_jobs=5)
        model = linear_model.LogisticRegression(max_iter=100000, C=0.1)

        # train (no batches)
        model.fit(self.partitions.x_train, self.partitions.y_train)

        # get regression coefficients
        self.coeff = model.coef_

        # prediction
        y_pred = model.predict(self.partitions.x_test)
        self.y_pred = y_pred

        return y_pred, self.partitions.y_test


class LogisticRegressionTuning:

    def __init__(self, partitions):
        self.partitions = partitions  # train and test data partitions
        self.y_pred = None
        self.best_params = None
        self.best_score = None

    def execute(self):
        # hyperparameter tuning
        param_grid = {'penalty': ['l1', 'l2'],
                      'C': np.linspace(0.001, 100, 50)}
        param_grid = {'penalty': ['l2'],
                      'C': [0.001]}

        # {'penalty': 'l2', 'C': 75.51044897959183}
        # Accuracy: 0.9
        # Precision score: 0.52
        # Recall score: 0.14
        # F1 score: 0.22

        # {'penalty': 'l2', 'C': 0.001}
        # Accuracy: 0.9
        # Precision score: 0.54
        # Recall score: 0.16
        # F1 score: 0.24


        # create model
        estimator = linear_model.LogisticRegression(max_iter=100000)
        search = RandomizedSearchCV(estimator, param_distributions=param_grid, n_iter=5, scoring='f1', n_jobs=-1, cv=5)
        search.fit(self.partitions.x_train, self.partitions.y_train)

        # prediction
        y_pred = search.predict(self.partitions.x_test)
        self.y_pred = y_pred
        self.best_score = search.best_score_
        self.best_params = search.best_params_

        return y_pred, self.partitions.y_test