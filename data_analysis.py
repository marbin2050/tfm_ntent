__author__ = '{Alfonso Aguado Bustillo}'

import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import pyplot
import numpy as np
from matplotlib import pylab


def check_for_nulls(data):

    # # empty values
    # empty_values = (df['text'] == ' ').value_counts()
    # print("\nEmpty values: " + str(empty_values))
    # # null values
    # null_values = df.isnull().sum().sum()
    # print("\nNull values: " + str(null_values))
    #
    # df.loc[df['text'] == ' ', 'text'] = 'empty text'
    # df['text'].fillna('empty text', inplace=True)
    #
    # empty_values = (df['text'] == ' ').value_counts()
    # print("\nEmpty values: " + str(empty_values))
    # # null values
    # null_values = df.isnull().sum().sum()
    # print("\nNull values: " + str(null_values))


    total = data.shape[0]
    for col in data.columns:
        zeroes = data[col].loc[data.loc[:, col].isin([0])].count()
        if zeroes == total:
            print(col + " only have zero values")


def h_approx(n):
    """
    Returns an approximate value of n-th harmonic number.
    http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + np.math.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)


def get_sub_types(data):

    n_is_redirect = data["is_redirect"].loc[data.loc[:, "is_redirect"].isin([1])].count()
    n_is_category_page = data["is_category_page"].loc[data.loc[:, "is_category_page"].isin([1])].count()
    n_is_category_redirect = data["is_category_redirect"].loc[data.loc[:, "is_category_redirect"].isin([1])].count()
    n_is_talkpage = data["is_talkpage"].loc[data.loc[:, "is_talkpage"].isin([1])].count()
    n_is_disambig = data["is_disambig"].loc[data.loc[:, "is_disambig"].isin([1])].count()
    n_is_filepage = data["is_filepage"].loc[data.loc[:, "is_filepage"].isin([1])].count()
    n_section = data["section"].loc[data.loc[:, "section"].isin([1])].count()

    sub_types = {'is_redirect': n_is_redirect, 'is_category_page': n_is_category_page,
                 'is_category_redirect': n_is_category_redirect, 'is_talkpage': n_is_talkpage,
                 'is_disambig': n_is_disambig, 'is_filepage': n_is_filepage, 'section': n_section}

    return sub_types


def show_pages_by_type(data, output_plots_path):

    sub_types = get_sub_types(data)
    sub_types = list(sub_types.values())
    n_sub_types = sum(sub_types)  # number of pages with subtype
    sub_types.append(n_sub_types)
    sub_types = sorted(sub_types, reverse=True)
    total_pages = data.shape[0]
    percentages = np.array(sub_types) / total_pages * 100  # % with respect to total pages (normal + subtype)
    n_bars = len(sub_types)  # nº plot bars

    ind_bars = np.arange(n_bars)  # the x bars
    bar_width = 0.6  # the width of the bars

    fig = plt.figure(figsize=(16, 16))
    # one bar per sub_type
    bars = plt.bar(ind_bars, tuple(percentages), bar_width, color='rgbkymcy')

    # place the % labels on the bars
    for rectangle, percentage in zip(bars, percentages):
        width = rectangle.get_width()
        height = rectangle.get_height()
        plt.text(rectangle.get_x() + width/5., rectangle.get_y() + height/1.5, str(round(percentage, 2)) + "%",
                 fontsize=18, color='black', weight='bold')

    plt.ylabel('% of pages')
    plt.title('Pages by subtype')
    plt.xticks(ind_bars, ('is_redirect', 'is_category_page', 'is_category_redirect', 'is_talkpage',
                          'is_disambig', 'is_filepage', 'section', 'any subtype'))

    fig.savefig(output_plots_path + 'page_subtype_ratios.png', dpi=300, bbox_inches='tight')
    pyplot.close(fig)


def word_frequency(data):
    word_freq = {}

    for index, value in data.items():
        for word in value:
            count = word_freq.get(word, 0)
            word_freq[word] = count + 1

    # show top 10 words
    # frequency = sorted(frequency.items(), key=lambda w: w[1], reverse=True)
    # frequency = {key: value for key, value in frequency.items()}

    return word_freq


def zip_law(word_freq, output_plots_path):

    freqs = sorted(word_freq.values(), reverse=True)
    n = len(freqs)
    ranks = range(1, n + 1)  # x-axis: the ranks
    # expected zipf
    k = sum(freqs) / h_approx(n)
    expected_freqs = [k / rank for rank in ranks]
    fig = pylab.figure(figsize=(10, 8))
    pylab.loglog(ranks, freqs, label='Zip Law Wikipedia')  # this plots frequency, not relative frequency
    pylab.loglog(ranks, expected_freqs,  label='Zip Law Expected')
    pylab.xlabel('log(rank)')
    pylab.ylabel('log(freq)')
    pylab.legend(loc='lower left')
    pylab.title('Wikipedia pages')
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

    ax = data_num.hist(figsize=(20, 20), bins=200, xlabelsize=8, ylabelsize=8)
    plt.figure()
    fig = ax[0][0].get_figure()
    fig.savefig(output_plots_path + 'numerical_vars_distribution.png', dpi=300, bbox_inches='tight')
    pyplot.close(fig)

    # STEP 4: STACKED BAR FOR BOOLEAN VARIABLES
    data_bool = data.select_dtypes(include=['boolean'])  # boolean data
    print(data_bool.head())
    # plot each page type in a separate column
    show_pages_by_type(data, output_plots_path)

    # STEP 5: PAGES WITH MORE THAN ONE SUBTYPE
    sub_types = ['is_redirect', 'is_category_page', 'is_category_redirect', 'is_talkpage', 'is_disambig',
                 'is_filepage', 'section']
    n_pages = (data.loc[:, sub_types].sum(axis=1) > 1).value_counts()[1]
    print("There are " + str(n_pages) + " pages with more than one subtype (" + str(round(n_pages/data.shape[0]*100))
          + "% of total pages)")

    # STEP 6: CORRELATION MATRIX
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

    # STEP 7: WORD FREQUENCY
    word_freq = word_frequency(text_list)

    # ZIP LAW
    zip_law(word_freq, output_plots_path)

    # STEP 9: WORD CLOUDS
