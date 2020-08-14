__author__ = '{Alfonso Aguado Bustillo}'

from feature_extraction import feature_extraction
from feature_selection import feature_selection
from load_pages import load_pages
from data_analysis import data_summary
from preprocessing import prepare_input_data
from learning_model import Partitions, LightGBM, DummyRegressor, \
    LinearRegressionBatch, LightGBMRFECV, LightGBMTuning
import evaluate

if __name__ == '__main__':

    # STEP 1: REQUEST AND STORE WIKIPEDIA PAGES
    # by executing the request_pages.py script

    # STEP 2: LOAD WIKIPEDIA PAGES
    pages_file = "data/output_files/_10000_top_docs_all.gz"
    data = load_pages(pages_file)

    # STEP 3: DATA SUMMARY OF PAGES LOADED
    # data_summary(data)

    # STEP 4: PREPROCESSING TRAINING/TEST DATA
    x, y_popularity, y_ranking = prepare_input_data(data)

    partitions = Partitions()
    partitions.create_data_partitions(x, y_popularity)

    # STEP 4: FEATURE EXTRACTION
    # feature_extraction(data)

    # STEP 5: FEATURE SELECTION
    # feature_selection(data)

    # STEP 6: Dummy regressor
    dr = DummyRegressor(partitions)
    y_pred, y_test = dr.execute()
    evaluate.summary(y_pred, y_test, "Dummy regressor [views]")

    # STEP 7: Linear regression
    lr = LinearRegressionBatch(partitions)
    y_pred, y_test = lr.execute()
    evaluate.summary(y_pred, y_test, "Linear Regression [views]")

    # STEP 8: LGBM
    # hyparameter tuning
    # lgbm_tuning = LightGBMTuning(partitions)
    # y_pred, y_test = lgbm_tuning.execute()
    # evaluate.summary(y_pred, y_test, "LightGBM tuning regression [views]")
    # best_params = lgbm_tuning.best_params  # get best params

    # model training
    lgbm = LightGBM(partitions, best_params=None)
    y_pred, y_test = lgbm.execute()
    evaluate.summary(y_pred, y_test, "LightGBM regression [views]")

    # recursive feature elimination to identify the best features (explain more variance)
    lgbm_rfecv = LightGBMRFECV(partitions)
    y_pred, y_test = lgbm_rfecv.execute()
    evaluate.summary(y_pred, y_test, "LightGBM RFECV regression [views]")

    # TODO: Empty texts are setting to length 1
    # TODO: Check what urls have text and title to null or empty
    # TODO: Remove stopwords from bag-of-words?
    # TODO: Finish check the null and empty values of data in data/preprocessing.py
    # TODO: Regex applied tp text and title in load_pages.py could be increasing the running time
    # TODO: Some texts are coming empty and I'm filling them with 'empty text' in preprocessing.py. Same for title
    # TODO: Change histograms by scatterplots
    # TODO: Add wordcloud by percentiles
    # TODO: Show message alerting if any of the variables have only null/zero values
