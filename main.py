__author__ = '{Alfonso Aguado Bustillo}'

from feature_extraction import feature_extraction
from feature_selection import feature_selection
from load_pages import load_pages
from data_analysis import data_wrangling, data_summary
from preprocessing import prepare_input_data
from learning_model import Partitions, LightGBM, DummyRegressor, \
    LinearRegressionBatch, LightGBMRFECV, LightGBMTuning, LightGBMBatch
import evaluate

if __name__ == '__main__':

    # STEP 1: REQUEST AND STORE WIKIPEDIA PAGES
    # by executing the request_pages.py script

    # STEP 2: LOAD WIKIPEDIA PAGES
    pages_file = "data/output_files/_75000_top_docs_all.gz"
    data = load_pages(pages_file)

    # data = data[:200]

    # STEP 3: DATA SUMMARY AND WRANGLING OF LOADED PAGES
    data_summary(data)
    data_wrangling(data)

    # # show y and log(y) distributions
    # import seaborn as sns
    # views = data['views']
    # sns.distplot(views, kde=False, rug=True)  # rug/ticks
    # sns.distplot(views, kde=True, rug=False)  # density
    # import numpy as np
    # log_views = np.log(data['views'])
    # sns.distplot(log_views, kde=False, rug=True)  # rug/ticks
    # sns.distplot(log_views, kde=True, rug=False)  # density
    #
    # # normality test
    # from scipy.stats import shapiro
    # stat, p = shapiro(views)
    # alpha = 0.05
    # if p > alpha:
    #     print('Sample looks Gaussian (fail to reject H0)')
    # else:
    #     print('Sample does not look Gaussian (reject H0)')
    #
    # stat, p = shapiro(log_views)
    # alpha = 0.05
    # if p > alpha:
    #     print('Sample looks Gaussian (fail to reject H0)')
    # else:
    #     print('Sample does not look Gaussian (reject H0)')

    # STEP 4: PREPROCESSING TRAINING/TEST DATA
    x, y_popularity, y_ranking = prepare_input_data(data)

    partitions = Partitions()
    partitions.create_data_partitions(x, y_popularity)

    # STEP 4: FEATURE EXTRACTION
    # feature_extraction(data)

    # STEP 5: FEATURE SELECTION
    # feature_selection(data)

    # STEP 6: Dummy regressor
    # dr = DummyRegressor(partitions)
    # y_pred, y_test = dr.execute()
    # evaluate.summary(y_pred, y_test, "Dummy regressor [views]")
    #
    # STEP 7: Linear regression
    # lr = LinearRegressionBatch(partitions)
    # y_pred, y_test = lr.execute()
    # evaluate.summary(y_pred, y_test, "Linear Regression [views]")

    # STEP 8: LGBM
    # hyparameter tuning
    # lgbm_tuning = LightGBMTuning(partitions)
    # y_pred, y_test = lgbm_tuning.execute()
    # evaluate.summary(y_pred, y_test, "LightGBM tuning regression [views]")
    # best_params = lgbm_tuning.best_params  # get best params

    # model training
    # lgbm = LightGBM(partitions, best_params=None)
    # y_pred, y_test = lgbm.execute()
    # evaluate.summary(y_pred, y_test, "LightGBM regression [views]")

    lgbm_batch = LightGBMBatch(partitions)
    y_pred, y_test = lgbm_batch.execute()
    evaluate.summary(y_pred, y_test, "LightGBM Batch regression [views]")

    # recursive feature elimination to identify the best features (explain more variance)
    # lgbm_rfecv = LightGBMRFECV(partitions)
    # y_pred, y_test = lgbm_rfecv.execute()
    # evaluate.summary(y_pred, y_test, "LightGBM RFECV regression [views]")

    # TODO: Empty texts are setting to length 1
    # TODO: Check what urls have text and title to null or empty
    # TODO: Remove stopwords from bag-of-words?
    # TODO: Finish check the null and empty values of data in data/preprocessing.py
    # TODO: Regex applied tp text and title in load_pages.py could be increasing the running time
    # TODO: Some texts are coming empty and I'm filling them with 'empty text' in preprocessing.py. Same for title
    # TODO: Change histograms by scatterplots
    # TODO: Add wordcloud by percentiles
    # TODO: Show message alerting if any of the variables have only null/zero values
