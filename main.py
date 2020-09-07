__author__ = '{Alfonso Aguado Bustillo}'

from load_pages import load_pages
from data_analysis import data_wrangling, data_summary
from preprocessing import prepare_x_features, prepare_y_clf_feature, prepare_y_reg_feature
from learning.graph_attention_network import GraphAttentionNetwork
from learning.learning_model import Partitions, LightGBM, DummyRegressor, LinearRegression, LogisticRegression
import evaluate
import numpy as np


def regression_problem(data, x_features):
    # y feature (dependent variable)
    # y feature (views/popularity) for regression
    y_reg = prepare_y_reg_feature(data)

    # regression partitions
    partitions = Partitions()
    partitions.create_data_partitions(x_features, y_reg, normalized=False, with_validation=False)

    # Linear Regression
    lr = LinearRegression(partitions)
    y_pred, y_test = lr.execute()
    evaluate.summary(y_pred, y_test, "Linear Regression")

    # LightGBM
    lgbm = LightGBM(partitions, best_params=None)
    y_pred, y_test = lgbm.execute()
    evaluate.summary(y_pred, y_test, "LightGBM regression")

    # Dummy Regressor
    dr = DummyRegressor(partitions)
    y_pred, y_test = dr.execute()
    evaluate.summary(y_pred, y_test, "Dummy regressor")


def classification_problem(data, x_features):
    lgr_results = {}
    gat_results = {}
    dummy_clf_results = {}

    # execute different buckets
    for majority_class_size in range(95, 55, -5):
        # y feature for classification
        y_clf = prepare_y_clf_feature(data, majority_class_size)

        partitions = Partitions()
        partitions.create_data_partitions(x_features, y_clf, normalized=False, with_validation=False)

        process_title = str(100 - majority_class_size) + "/" + str(majority_class_size) + "]"
        key_name = str(100 - majority_class_size) + "/" + str(majority_class_size)
        # Logistic Regression
        lgr = LogisticRegression(partitions)
        y_pred, y_test = lgr.execute()

        title = "Logistic regression [" + process_title
        lgr_result = evaluate.summary(y_pred, y_test, regression=False, process_title=title)
        lgr_results[key_name] = lgr_result

        # Graph Attention Network
        gat = GraphAttentionNetwork(data, x_features, y_clf)
        y_pred, y_test = gat.execute()
        title = "GAT [" + process_title
        gat_result = evaluate.summary(y_pred, y_test, regression=False, process_title=title)
        gat_results[key_name] = gat_result

        # Dummy classifier
        y_pred_dummy = [0] * len(partitions.y_test)
        title = "Dummy Classifier [" + process_title
        dummy_clf_result = evaluate.summary(y_pred_dummy, partitions.y_test, regression=False, process_title=title)
        dummy_clf_results[key_name] = dummy_clf_result

    # show metric's plots for Logistic Regression
    evaluate.plots_summary(lgr_results, output_file_name='data/output_files/lgr_results.png')
    # show metric's plots for Graph Attention Network
    evaluate.plots_summary(gat_results, output_file_name='data/output_files/gat_results.png')
    # show metric's plots for Dummy Classifier
    evaluate.plots_summary(dummy_clf_results, output_file_name='data/output_files/dummy_clf_results.png')


def main():

    # REQUEST AND STORE WIKIPEDIA PAGES
    # by executing the request_pages.py script

    # LOAD WIKIPEDIA PAGESº
    pages_file = "data/output_files/_75000_top_docs_all.gz"
    data = load_pages(pages_file)
    # dataset of 10.000 samples
    data = data.sample(n=50000, random_state=42)
    data = data.sort_values('views', ascending=False).reset_index(drop=True)

    # add views (log) to the dataframe
    views_log = np.log(data['views'])
    data.insert(1, 'views_log', views_log)

    # STEP 3: DATA SUMMARY AND WRANGLING OF LOADED PAGES
    data = data_wrangling(data)
    data_summary(data)

    # extract the x features or independent variables
    x = prepare_x_features(data)

    # REGRESSION PROBLEM
    regression_problem(data, x)

    # CLASSIFICATION PROBLEM
    # classification_problem(data, x)


main()
