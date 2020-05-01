__author__ = '{Alfonso Aguado Bustillo}'

import pandas as pd


def extract_values(all_queries, column_names, column_indexes):

    # get only the columns that are document features (static ranking)
    all_queries = all_queries[column_indexes]

    row_list = []
    for row_tuple in all_queries.itertuples(index=False):
        relevance = row_tuple[0]  # relevance is the first column
        row = {column_names[0]: relevance}
        # extract values
        for column_name, row_value in zip(column_names[1:], row_tuple[1:]):
            row[column_name] = row_value.split(":", 1)[1]

        row_list.append(row)

    # dataframe with only valid columns and values
    all_queries = pd.DataFrame(row_list, columns=column_names)

    return all_queries
