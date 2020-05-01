__author__ = '{Alfonso Aguado Bustillo}'

import pandas as pd
import documents
import preprocessing
import graph
import learn_to_rank
import evaluate


# COMMAND LINE ARGUMENTS
fold_file_1 = "input_files/train_sample.txt"
column_indexes = [0, 1, 12, 13, 14, 15, 16, 127, 128, 129, 130, 131, 132, 133, 134, 136, 137]
column_names = ['relevance', 'qid', 's_length_body', 's_length_anchor', 's_length_title', 's_length_url',
                's_length_document', 'n_slash_url', 'length_url', 'inlink_number', 'outlink_number',
                'pagerank', 'siterank', 'quality_score', 'quality_score2', 'url_click_count', 'url_dwell_time']
# LightGBM training parameters
hyperparams = {'n_iter': 10000, 'learning_rate': 0.00003, 'boosting_type': 'gbdt', 'objective': 'regression',
               'metric': 'mae', 'sub_feature': 0.5, 'num_leaves': 10, 'min_data': 50, 'max_depth': 10}


def main():

    # STEP 1: GET DOCUMENTS FROM QUERIES
    # load the queries file
    all_queries = pd.read_csv(fold_file_1, sep=" ", header=None)
    # pre-processing: format and extract values from queries
    all_queries = preprocessing.extract_values(all_queries, column_names, column_indexes)
    # create documents
    docs_sorted = documents.get_documents(all_queries)

    # STEP 2: CREATE GRAPH
    # get documents by query
    docs_by_query = documents.get_documents_by_query(docs_sorted)
    graph.create_graph(docs_by_query)

    # STEP 3: PREDICTION and EVALUATION
    docs_sorted = docs_sorted.drop(['qid'], axis=1)  # drop not necessary qid column
    # get input and output values for the algorithms
    x = docs_sorted[column_names[2:]].values  # get input values
    y = docs_sorted['relevance'].values  # get output values
    partitions = learn_to_rank.Partitions()
    partitions.create_data_partitions(x, y)  # train and test data for the algorithms

    # execute linear regression
    lr = learn_to_rank.LinearRegression(partitions)
    y_pred, y_test = lr.execute()  # run algorithm
    evaluate.summary(y_pred, y_test)  # evaluate

    # execute light gbm on regression
    lgbm = learn_to_rank.LightGBM(partitions, hyperparams)
    y_pred, y_test =  lgbm.execute()
    evaluate.summary(y_pred, y_test)

    # execute lambda rank
    lrank = learn_to_rank.LambdaRank(partitions)
    y_pred, y_test = lrank.execute()
    evaluate.summary(y_pred, y_test)


main()



