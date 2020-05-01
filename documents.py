__author__ = '{Alfonso Aguado Bustillo}'

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as seabornInstance


def get_documents(all_queries):

    # get column names
    column_names = all_queries.columns

    # select the features only, as that identifies our document, and then compute the max
    docs = all_queries.groupby(['{}'.format(fid) for fid in column_names[2:]]).agg({'qid': list, 'relevance': max})

    # reset the index to end up with a dataframe
    doc_labels = docs.reset_index()

    # create the document id column, by simply setting the index of the dataframe
    doc_labels['doc_id'] = doc_labels.index

    # sort by the label -- this is not necessary, but it helps you understand the data
    docs_sorted = doc_labels.sort_values('relevance')

    return docs_sorted


def get_documents_by_query(documents):

    # create the mapping from query IDs to documents
    query_documents = defaultdict(set)
    for index, row in documents.iterrows():
        for qid in row.qid:
            query_documents[qid].add(row.doc_id)

    return query_documents


def summary(documents, column_name):

    # Take a look at the dataframe
    print(documents.head())

    # show distribution by column_name
    plt.figure(figsize=(15, 10))
    plt.tight_layout()
    seabornInstance.distplot(documents[column_name])
