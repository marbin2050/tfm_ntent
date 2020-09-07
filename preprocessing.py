from collections import defaultdict

import nltk

__author__ = '{Alfonso Aguado Bustillo}'

from scipy.sparse import hstack, vstack
from scipy.sparse import csr_matrix
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re
from itertools import combinations
from igraph import *
from nltk.tokenize.treebank import TreebankWordDetokenizer
from ordered_set import OrderedSet
import numpy as np


class BagOfWords:

    def __init__(self, original_text_list):
        self.original_text_list = original_text_list

    @staticmethod
    def remove_white_spaces(text):
        # strip whitespaces
        return text.strip()

    @staticmethod
    def remove_punctuation(text):
        # create a dictionary of punctuation characters
        punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                                    if unicodedata.category(chr(i)).startswith('P'))
        return text.translate(punctuation)

    @staticmethod
    def remove_non_alphabetical(text):
        return re.sub(r'\W+', ' ', re.sub(r'\d+', ' ', text))  # remove non-alphabetical characters via regex

    @staticmethod
    def tokenize(text):
        # tokenize words
        tokenized_text = word_tokenize(text)
        tokenized_text = list(set(tokenized_text))  # unique values
        return tokenized_text

    @staticmethod
    def detokenize(word_list):
        # detokenize words
        sentence = TreebankWordDetokenizer().detokenize(word_list)
        return sentence

    @staticmethod
    def remove_stop_words(text):
        # load stop words
        stop_words = stopwords.words('english')
        # remove stop words
        new_text = []
        for word in text:
            word = word.lower()
            if word not in stop_words:
                new_text.append(word)
        return new_text

    @staticmethod
    def remove_wikipedia_words(text):
        # load stop words
        wikipedia_words = ['url', 'infobox', 'imagename', 'imagesize', 'plainlist', 'mf', 'ref', 'http', 'row', 'cite',
                           'urlstatus', 'accessdate', 'imagecaption', 'div', 'b', 'authorlink', 'vii', 'vi', 'viii',
                           'date', 'name', 'title', 'image', 'short', 'description', 'caption', 'br', 'us', 'dmy',
                           'archivedate', 'httpswebarchiveorgweb', 'archiveurl', 'p', 'c', 'df', 'th', 'type', 'px',
                           'mdy', 'html', 'z', 'x', 'tg']

        # remove stop words
        new_text = []
        for word in text:
            word = word.lower()
            if word not in wikipedia_words:
                new_text.append(word)
        return new_text

    @staticmethod
    def stemm_words(text):
        # create stemmer
        porter = PorterStemmer()
        # apply stemmer
        new_text = []
        for word in text:
            new_text.append(porter.stem(word))
        return new_text

    @staticmethod
    def encode_text(text):
        # create the bag of words feature matrix
        count = CountVectorizer()
        bag_of_words = count.fit_transform(text)
        return bag_of_words

    def execute(self):
        original_texts_list = self.original_text_list
        new_text_list = []

        for original_text in original_texts_list:
            try:
                text = original_text
                text = self.remove_white_spaces(text)
                text = self.remove_punctuation(text)
                text = self.remove_non_alphabetical(text)
                word_list = self.tokenize(text)
                word_list = self.remove_stop_words(word_list)
                word_list = self.remove_wikipedia_words(word_list)
                new_text = self.detokenize(word_list)
                new_text_list.append(new_text)

            except Exception as e:
                print("Exception: " + str(e))
                print("No words found in intro.\n")
                pass

        bag_of_words = self.encode_text(new_text_list)
        return bag_of_words


def create_graph(data, only_main_pages=False):

    vertices = OrderedSet()
    edges = OrderedSet()

    # get the main urls/nodes
    main_pages_url = OrderedSet([full_url[full_url.rfind('/') + 1:] for full_url in data['full_url']])

    for index, row in data.iterrows():
        url = row.full_url[row.full_url.rfind('/') + 1:]  # get last part of url to match internal links format
        vertices.add(url)
        for internal_link in row.internal_links:
            if only_main_pages:
                if internal_link in main_pages_url:
                    edges.add((url, internal_link))
                    # vertices.add(internal_link)
            else:
                edges.add((url, internal_link))
                vertices.add(internal_link)

    # create graph
    graph = Graph(directed=True)
    # add vertices
    graph.add_vertices(list(vertices))
    # add edges
    graph.add_edges(list(edges))

    # get index of the vertices (main pages)
    index_list = []
    for url in list(main_pages_url):
        index_list.append(graph.vs._name_index.get(url))

    return graph, index_list


def prepare_graph_features(data):

    graph, index_list = create_graph(data)

    # in-degree
    indegree = graph.indegree()
    indegree = [indegree[i] for i in index_list]
    data.insert(len(data.columns), 'indegree', indegree)
    # out-degree
    outdegree = graph.outdegree()
    outdegree = [outdegree[i] for i in index_list]
    data.insert(len(data.columns), 'outdegree', outdegree)
    # pagerank
    pagerank = graph.pagerank()
    pagerank = [pagerank[i] for i in index_list]
    data.insert(len(data.columns), 'pagerank', pagerank)
    # authority score
    authority_score = graph.authority_score()
    authority_score = [authority_score[i] for i in index_list]
    data.insert(len(data.columns), 'authority_score', authority_score)
    # hub score
    hub_score = graph.hub_score()
    hub_score = [hub_score[i] for i in index_list]
    data.insert(len(data.columns), 'hub_score', hub_score)

    graph_features = data.loc[:, ['indegree', 'outdegree', 'pagerank', 'authority_score', 'hub_score']]
    graph_features = graph_features.astype('f4')

    return graph_features


def bag_of_words_intro(data):

    # bag of words for the introduction text
    bow_intro = BagOfWords(data.loc[:, 'introduction_text'])
    intro_words_encoded = bow_intro.execute()
    intro_words_encoded = csr_matrix(intro_words_encoded)
    intro_words_encoded = intro_words_encoded.astype('f4')

    return intro_words_encoded


def bag_of_words_categories(data):

    bow_categories = BagOfWords(data)
    new_categories = []

    for category_list in data['category_names']:
        category_list = [category[category.find("%")+1:] for category in category_list]
        new_text = bow_categories.detokenize(category_list)
        new_categories.append(new_text)

    categories_encoded = bow_categories.encode_text(new_categories)
    category_features = csr_matrix(categories_encoded)
    category_features = category_features.astype('f4')

    return category_features


def prepare_text_features(data):

    # bag of words introduction text
    # bow_intro = bag_of_words_intro(data)

    text_features = data.loc[:, ['title_length', 'n_title_words', 'n_introduction_words', 'n_full_text_words',
                                 'bytes_introduction_text', 'bytes_full_text']].values

    text_features = csr_matrix(text_features)
    # text_features = hstack([text_features, bow_intro])

    text_features = text_features.astype('f4')

    return text_features


def prepare_link_features(data):
    link_features = data.loc[:, ['n_citations', 'n_external_links', 'n_image_links',
                                 'n_lang_links', 'n_internal_links']].values

    link_features = link_features.astype('f4')

    return link_features


def prepare_contributor_features(data):
    contributor_features = data.loc[:, ['n_contributors', 'n_contributors_edits']].values
    contributor_features = contributor_features.astype('f4')

    return contributor_features


def prepare_temporal_features(data):
    temporal_features = data.loc[:, ['time_since_last_edit', 'time_since_first_edit']].values

    return temporal_features


def prepare_page_type_features(data):

    # bag of words categories
    # bow_category = bag_of_words_categories(data)

    infobox_features = prepare_infobox_features(data)

    page_type_features = data.loc[:, ['n_categories', 'n_sections',
                                      'is_category_page', 'is_disambig', 'is_talkpage', 'is_filepage']].values

    page_type_features = page_type_features.astype('f4')

    # page_type_features = hstack([bow_category, infobox_features, page_type_features])
    page_type_features = hstack([infobox_features, page_type_features])

    return page_type_features


def prepare_infobox_features(data):

    infoboxes = []
    for infobox_list in data['infoboxes']:
        if infobox_list:
            infoboxes.append(1)
        else:
            infoboxes.append(0)

    infoboxes = csr_matrix(infoboxes).transpose()
    infoboxes = infoboxes.astype('f4')

    return infoboxes


def prepare_x_features(data):
    # sort data views
    # data = data.sort_values('views', ascending=False)

    # x features (independent variables)
    # text features
    text_features = prepare_text_features(data)
    text_features = csr_matrix(text_features)  # csr matrix

    # link features
    link_features = prepare_link_features(data)
    link_features = csr_matrix(link_features)

    # contributor features
    contributor_features = prepare_contributor_features(data)
    contributor_features = csr_matrix(contributor_features)

    # page type features
    page_type_features = prepare_page_type_features(data)
    page_type_features = csr_matrix(page_type_features)

    # graph features
    graph_features = prepare_graph_features(data)
    graph_features = csr_matrix(graph_features)

    # join all x features for the algorithms
    x_features = hstack([text_features, link_features, page_type_features, contributor_features, graph_features])

    # convert to csr matrix (hstack returns a coo sparse matrix)
    # to allow slicing over this matrix when training the algorithms (mini-batching)
    x_features = x_features.tocsr()

    return x_features


def prepare_y_clf_feature(data, majority_class_size):

    y_clf = data.loc[:, ['views_log']].values
    perc_up = np.percentile(y_clf, majority_class_size)
    encoded_y = []
    # for views in y_reg:
    for views in y_clf:
        if views > perc_up:
            encoded_y.append(1)
        else:
            encoded_y.append(0)
    y_clf = np.array(encoded_y)

    return y_clf


def prepare_y_reg_feature(data):
    y = data.loc[:, ['views_log']].values

    return y