__author__ = '{Alfonso Aguado Bustillo}'

import matplotlib.pyplot as plt
import pandas as pd


def check_for_nulls(data):
    total = data.shape[0]
    for col in data.columns:
        zeroes = data[col].loc[data.loc[:, col].isin([0])].count()
        if zeroes == total:
            print(col + " only have zero values")


def get_pages_type(data):

    ones1 = data["is_redirect"].loc[data.loc[:, "is_redirect"].isin([1])].count()
    ones2 = data["is_category_page"].loc[data.loc[:, "is_category_page"].isin([1])].count()
    ones3 = data["is_category_redirect"].loc[data.loc[:, "is_category_redirect"].isin([1])].count()
    ones4 = data["is_talkpage"].loc[data.loc[:, "is_talkpage"].isin([1])].count()
    ones5 = data["is_disambig"].loc[data.loc[:, "is_disambig"].isin([1])].count()
    ones6 = data["is_filepage"].loc[data.loc[:, "is_filepage"].isin([1])].count()
    ones7 = data["section"].loc[data.loc[:, "section"].isin([1])].count()
    # total = data.shape[0]
    # ones8 = total-ones1-ones2-ones3-ones4-ones5-ones6-ones7  # normal pages?
    all_classes = {'is_redirect': ones1, 'is_category_page': ones2, 'is_category_redirect': ones3, 'is_talkpage': ones4,
                   'is_disambig': ones5, 'is_filepage': ones6, 'section': ones7}

    return all_classes


def data_summary(data):

    # STEP 1: FIRST LOOK AT DATA
    print(data.head())
    print(data.info())

    # STEP 2: CHECK NULL (ZERO) VALUES
    check_for_nulls(data)

    # STEP 2: DESCRIPTIVE STATISTICS OF VARIABLES
    print(data.describe())

    # STEP 3: HISTOGRAMS FOR EACH NUMERICAL VARIABLE (DISTRIBUTIONS)
    data_num = data.select_dtypes(include=['int64'])  # numerical data
    print(data_num.head())
    data_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
    plt.show()

    # STEP 4: STACKED BAR FOR BOOLEAN VARIABLES
    data_bool = data.select_dtypes(include=['boolean'])  # boolean data
    print(data_bool.head())
    df = pd.DataFrame(get_pages_type(data_bool), index=[0])
    df.plot.bar(stacked=True)

    # TODO: show message alerting if any of the variables have only null/zero values
