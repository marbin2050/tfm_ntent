__author__ = '{Alfonso Aguado Bustillo}'

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from matplotlib import pyplot
import numpy as np


def normalize(val, max, min):
    return (val - min) / (max - min) * 100


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
    all_classes = {'is_redirect': ones1, 'is_category_page': ones2, 'is_category_redirect': ones3, 'is_talkpage': ones4,
                   'is_disambig': ones5, 'is_filepage': ones6, 'section': ones7}

    return all_classes


def show_pages_by_type(data, output_plots_path):

    total = data.sum(1)[0]
    n_columns = data.shape[1]  # number of bars
    sub_types = data.values.tolist()[0]

    # normalize to 0-100 range
    sub_types_norm = []
    for i in sub_types:
        sub_types_norm.append(normalize(i, total, 0))
    diff_norm = [100 - element for element in sub_types_norm]

    indices = np.arange(n_columns)  # the x locations for the groups
    bar_width = 0.6  # the width of the bars

    fig = plt.figure(figsize=(16, 16))
    # two bars/rectangles at position of the x axis
    p1 = plt.bar(indices, tuple(sub_types_norm), bar_width)
    p2 = plt.bar(indices, tuple(diff_norm), bar_width, bottom=sub_types_norm)

    # needed for placing the % over the bars
    for rectangle in p1:
        width = rectangle.get_width()
        height = rectangle.get_height()
        plt.text(rectangle.get_x() + width/5., rectangle.get_y() + height/1.5, str(round(height)) + "%", fontsize=18,
                 color='white', weight='bold')

    plt.ylabel('% of pages')
    plt.title('Pages by type')
    plt.xticks(indices, ('is_redirect', 'is_category_page', 'is_category_redirect',
                         'is_talkpage', 'is_disambig', 'is_filepage', 'section'))
    plt.legend((p1[0], p2[0]), ('Type', 'Total'))

    fig.savefig(output_plots_path + 'page_type_ratios.png', dpi=300, bbox_inches='tight')
    pyplot.close(fig)


def word_frequency(data, output_plots_path):
    frequency = {}

    for index, value in data.items():
        for word in value:
            count = frequency.get(word, 0)
            frequency[word] = count + 1

    # show top 10 words
    # frequency = sorted(frequency.items(), key=lambda w: w[1], reverse=True)
    #frequency = {key: value for key, value in frequency.items()}

    # plot zip law for top 1000
    n_bars = np.arange(1000)
    fig = plt.figure(figsize=(10, 8))
    # s = 1
    # expected_zipf = [term_freq_df.sort_values(by='total', ascending=False)['total'][0] / (i + 1) ** s for i in y_pos]
    plt.bar(n_bars, sorted(frequency.values(), reverse=True)[0:1000], align='center', alpha=0.5)
    # plt.plot(n_bars, expected_zipf, color='r', linestyle='--', linewidth=2, alpha=0.5)
    plt.ylabel('Frequency')
    plt.title('Top 1000 words in Wikipedia pages')
    fig.savefig(output_plots_path + 'zip_law.png', dpi=300, bbox_inches='tight')
    pyplot.close(fig)


def data_summary(data):

    output_plots_path = "data/output_files/data_analysis/"

    # get text column
    text_list = data.loc[:, "text"]

    # DROP text, text_features and links OUT while they are not received (request Wikipedia data again)
    data = data.drop(['text', 'text_features', 'links'], axis=1)

    # STEP 1: FIRST LOOK AT DATA
    print(data.head())
    print(data.info())

    # STEP 2: CHECK ALL NULL (ZERO) VALUES
    check_for_nulls(data)

    # STEP 2: DESCRIPTIVE STATISTICS OF VARIABLES
    print(data.describe())

    # STEP 3: HISTOGRAMS FOR EACH NUMERICAL VARIABLE (DISTRIBUTIONS)
    data_num = data.select_dtypes(include=['int64'])  # numerical data
    print(data_num.head())
    # plot
    ax = data_num.hist(figsize=(20, 20), bins=50, xlabelsize=8, ylabelsize=8)
    plt.figure()
    fig = ax[0][0].get_figure()
    fig.savefig(output_plots_path + 'numerical_vars_distribution.png', dpi=300, bbox_inches='tight')
    pyplot.close(fig)

    # STEP 4: STACKED BAR FOR BOOLEAN VARIABLES
    data_bool = data.select_dtypes(include=['boolean'])  # boolean data
    print(data_bool.head())
    # plot
    df = pd.DataFrame(get_pages_type(data_bool), index=[0])
    # plot each page type in a separate column
    show_pages_by_type(df, output_plots_path)
    # all page types in one stack
    ax = df.plot.bar(stacked=True)
    plt.figure()
    fig = ax.get_figure()
    fig.savefig(output_plots_path + 'page_type_ratios_one_bar.png', dpi=300, bbox_inches='tight')
    pyplot.close(fig)

    # STEP 5: CORRELATION MATRIX
    corr_matrix = data_num.corr()
    # correlations sorted in descending order
    corr = corr_matrix.stack().abs()
    corr = corr[corr.index.get_level_values(0) != corr.index.get_level_values(1)]
    corr = corr.sort_values(ascending=False)
    corr = corr.drop_duplicates()
    print(corr)
    # plot matrix
    ax = sn.heatmap(corr_matrix, annot=True, annot_kws={"fontsize":6})
    plt.figure(figsize=(30, 30))
    fig = ax.get_figure()
    fig.savefig(output_plots_path + 'num_vars_corr_matrix.png', dpi=300, bbox_inches='tight')
    pyplot.close(fig)

    # STEP 6: WORD FREQUENCY
    word_frequency(text_list, output_plots_path)
