__author__ = '{Alfonso Aguado Bustillo}'

from data.load_pages import load_pages
from data.data_analysis import data_summary


if __name__ == '__main__':

    # STEP 1: REQUEST AND STORE WIKIPEDIA PAGES
    # by executing the request_pages.py script

    # STEP 2: LOAD WIKIPEDIA PAGES
    pages_file = "data/output_files/top_docs_2140.jsonl.gz"
    # pages_file = "data/output_files/top_docs.gz"
    data = load_pages(pages_file)

    # STEP 3: DATA SUMMARY OF PAGES LOADED
    data_summary(data)

    # TODO: Graph for zip law crashes when more than 1-2K bars
    # TODO: Add the expected zip law and maybe delete outliers (highest ones)
    # TODO: Add wordcloud by percentiles
    # TODO: Check how many pages 'have' more than one page type
    # TODO: show message alerting if any of the variables have only null/zero values
