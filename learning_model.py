import sklearn.linear_model as linear_model
from lightgbm import LGBMRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV
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

    def create_data_partitions(self, x, y):

        # split in train and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        # split train and validation data
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train,
                                                                              test_size=0.25, random_state=0)
        # log dependent variable
        self.y_train = np.log(self.y_train)
        self.y_val = np.log(self.y_val)
        self.y_test = np.log(self.y_test)

        # feature Scaling
        sc_x = StandardScaler(with_mean=False)
        sc_y = StandardScaler(with_mean=False)
        self.x_train = sc_x.fit_transform(self.x_train)
        self.y_train = sc_y.fit_transform(self.y_train)
        self.x_val = sc_x.fit_transform(self.x_val)
        self.y_val = sc_y.fit_transform(self.y_val)
        self.x_test = sc_x.fit_transform(self.x_test)
        self.y_test = sc_y.fit_transform(self.y_test)


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


class LinearRegressionBatch:

    def __init__(self, partitions):
        self.partitions = partitions  # train and test data partitions
        self.coeff = None
        self.y_pred = None
        self.score = None

    def execute(self):

        # create model
        model = linear_model.SGDRegressor()

        # indexes to be used for getting minibatches
        indexes = np.arange(self.partitions.x_train.shape[0])

        while len(indexes) > 0:
            # get minibatches
            x_train, y_train, indexes = get_minibatch(self.partitions.x_train, self.partitions.y_train, indexes,
                                                      size=100)

            model.partial_fit(x_train, y_train)

        # train (no batches)
        # model.fit(self.partitions.x_train, self.partitions.y_train)

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
        self.params = {'n_iter': 10000, 'learning_rate': 0.001, 'boosting_type': 'gbdt', 'objective': 'regression',
                       'metric': 'mae', 'num_leaves': 350, 'max_depth': 9, 'min_data_in_leaf': 100,
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
        self.params = {'n_iter': 10000, 'learning_rate': 0.001, 'boosting_type': 'gbdt', 'objective': 'regression',
                       'metric': 'mae', 'num_leaves': 350, 'max_depth': 9, 'min_data_in_leaf': 100,
                       'verbose': -1}

        self.partitions = partitions  # train and test data partitions
        self.y_pred = None

    def execute(self):

        size = 1000
        estimator = None
        for iteration, x in enumerate(range(0, self.partitions.x_train.shape[0] - size, size)):
            indices = list(range(x, x + size))

            estimator = lgb.train(self.hyperparams,
                                  init_model=estimator,
                                  train_set=lgb.Dataset(self.partitions.x_train[indices].toarray(),
                                                        self.partitions.y_train[indices].flatten()),
                                  keep_training_booster=True,
                                  num_boost_round=100)

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
                      'max_depth': [9, 10, 12],
                      'learning_rate': [0.001, 0.0001],
                      'min_data_in_leaf': [50, 75, 100],
                      'max_bin': [100, 255, 500],
                      'n_iter': [10000]}

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

        # select only the best features
        x_test = self.partitions.x_test[:, selector.support_]
        # prediction
        y_pred = selector.predict(x_test)

        # number of best features
        self.n_features = selector.n_features_

        # which categories are best
        self.best_features = selector.support_

        # rank features best (1) to worst
        self.feature_ranking = selector.ranking_

        return y_pred, self.partitions.y_test


# class LightGBMRFECVTuning:
#
#     def __init__(self, partitions):
#         self.partitions = partitions  # train and test data partitions
#
#     def execute(self):
#
#         param_grid = [{'estimator__num_leaves': [70]},
#                       {'estimator__max_depth': [7]}]
#
#         # create model
#         estimator = LGBMRegressor(boosting_type='gbdt', objective='regression')
#
#         selector = RFECV(estimator, step=1, cv=4, scoring="neg_mean_squared_error")
#         search = GridSearchCV(selector, param_grid, cv=7)
#         search.fit(self.partitions.x_train, self.partitions.y_train)
#         search.best_estimator_.estimator_
#         search.best_estimator_.grid_scores_
#         search.best_estimator_.ranking_
#
#         # select only the best features
#         x_test = self.partitions.x_test[:, search.best_estimator_.support_]
#
#         # prediction
#         y_pred = search.best_estimator_.estimator_.predict(x_test)
#         self.y_pred = y_pred
#
#         return y_pred, self.partitions.y_test

