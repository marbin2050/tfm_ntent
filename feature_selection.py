__author__ = '{Alfonso Aguado Bustillo}'

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def text_features_selection(data):

    text_features = data.loc[:, ['title_length', 'n_title_words', 'n_introduction_words', 'n_full_text_words',
                                 'bytes_introduction_text', 'bytes_full_text', 'n_citations', 'n_sections']].values

    # VARIANCE
    # Create thresholder
    thresholder = VarianceThreshold(threshold=.5)
    # Create high variance feature matrix
    features_high_variance = thresholder.fit_transform(text_features)
    # View variances
    thresholder.fit(text_features).variances_
    indices = thresholder.get_support()
    # ACABAR!!! Habría que eliminar aquellas text features que no superan el thresold

    # HIGHLY CORRELATED FEATURES
    # Convert feature matrix into DataFrame
    dataframe = pd.DataFrame(text_features)
    # Create correlation matrix
    corr_matrix = dataframe.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                      k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    # Drop features
    dataframe.drop(dataframe.columns[to_drop], axis=1).head(3)
    # ACABAR Y QUE ESTÉ CLARO QUÉ COLUMNA ES CADA UNA

    # RECURSIVE ELIMINATING FEATURES
    import warnings
    from sklearn.datasets import make_regression
    from sklearn.feature_selection import RFECV
    from sklearn import linear_model
    # Suppress an annoying but harmless warning
    warnings.filterwarnings(action="ignore", module="scipy",
                            message="^internal gelsd")

    # Create a linear regression
    ols = linear_model.LinearRegression()
    # Recursively eliminate features
    rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")
    target = data.loc[:, ['views']].values
    rfecv.fit(text_features, target)
    rfecv.transform(text_features)
    # Number of best features
    rfecv.n_features_
    # Which categories are best
    rfecv.support_
    # Rank features best (1) to worst
    rfecv.ranking_


def feature_selection(data):

    text_features_selection(data)
