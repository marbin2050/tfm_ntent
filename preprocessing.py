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

                # word_list = list(set(word_list))  # unique values
                # # nltk.download('averaged_perceptron_tagger')
                # word_list_tagged = pos_tag(word_list)
                # # Filter words
                # new_word_list_tagged = []
                # for word, tag in word_list_tagged:
                #     if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                #         new_word_list_tagged.append(word)
                # new_text = self.detokenize(new_word_list_tagged)
                # new_text_list.append(new_text)

            except Exception as e:
                print("Exception: " + str(e))
                print("No words found in intro.\n")
                pass

        bag_of_words = self.encode_text(new_text_list)
        return bag_of_words


def create_graph(pages_mapping):

    # create the set of edges -- ideally you would dump this to a file rather than keep it in memory
    edges = set()
    for page_link in pages_mapping.values():
        edges.update(set(combinations(page_link, 2)))

    # create graph
    graph = Graph()

    # add nodes
    for page_node in pages_mapping.keys():
        graph.add_node(page_node)

    # add edges
    for edge in edges:
        graph.add_edge(edge[0], edge[1])


def prepare_graph_features(data):

    vertices = []
    edges = []
    main_pages_url = []

    for index, row in data.iterrows():
        url = row.full_url[row.full_url.rfind('/') + 1:]  # get last part of url to match internal links format
        main_pages_url.append(url)
        vertices.append(url)
        for internal_link in row.internal_links:
            edges.append((url, internal_link))
            vertices.append(internal_link)

        # for back_link in row.back_links:
        #     edges.append((back_link, url))
        #     vertices.append(back_link)

    # create graph
    graph = Graph(directed=True)
    # add vertices
    graph.add_vertices(vertices)
    # add edges
    graph.add_edges(edges)

    # get index of the vertices (main pages)
    index_list = []
    for url in main_pages_url:
        index_list.append(graph.vs._name_index.get(url))

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

    return graph_features


def bag_of_words_intro(data):

    # bag of words for the introduction text
    bow_intro = BagOfWords(data.loc[:, 'introduction_text'])
    intro_words_encoded = bow_intro.execute()

    return intro_words_encoded


def bag_of_words_categories(data):

    bow_categories = BagOfWords(data)
    new_categories = []

    for category_list in data['category_names']:
        category_list = [category[category.find("%")+1:] for category in category_list]
        new_text = bow_categories.detokenize(category_list)
        new_categories.append(new_text)

    categories_encoded = bow_categories.encode_text(new_categories)

    return categories_encoded


def prepare_text_features(data):

    # bag of words introduction text
    # intro_words_encoded = bag_of_words_intro(data)

    text_features = data.loc[:, ['title_length', 'n_title_words', 'n_introduction_words', 'n_full_text_words',
                                 'bytes_introduction_text', 'bytes_full_text', 'n_citations', 'n_sections']].values

    text_features = csr_matrix(text_features)
    # text_features = hstack([text_features, intro_words_encoded])

    return text_features


def prepare_link_features(data):
    link_features = data.loc[:, ['n_internal_links', 'n_external_links', 'n_image_links',
                                 'n_back_links', 'n_lang_links']].values

    return link_features


def prepare_contributor_features(data):
    contributor_features = data.loc[:, ['n_contributors', 'n_contributors_edits', 'time_since_last_edit',
                                        'time_since_first_edit']].values

    return contributor_features


def prepare_category_features(data):

    # bag of words categories
    # categories_encoded = bag_of_words_categories(data)

    category_features = data.loc[:, ['n_categories']].values
    category_features = csr_matrix(category_features)
    # category_features = hstack([category_features, categories_encoded])

    return category_features


def prepare_page_type_features(data):
    page_type_features = data.loc[:, ['is_category_page', 'is_disambig', 'is_talkpage', 'is_filepage']].values
    return page_type_features


def prepare_infobox_features(data):

    infoboxes = []
    for infobox_list in data['infoboxes']:
        if infobox_list:
            infoboxes.append(1)
        else:
            infoboxes.append(0)

    return infoboxes


def prepare_input_data(data):
    # sort data views
    data = data.sort_values('views', ascending=False)

    # STEP 1: y features (dependent variables)
    # y feature (views/popularity)
    y_popularity = data.loc[:, ['views']].values
    # y_popularity = y_popularity.flatten()
    # y feature (ranking/order)
    y_ranking = data['views'].rank(method='dense')

    # STEP 2: x features (independent variables)
    # text features
    text_features = prepare_text_features(data)
    text_features = csr_matrix(text_features)  # csr matrix

    # link features
    link_features = prepare_link_features(data)
    link_features = csr_matrix(link_features)

    # contributor features
    contributor_features = prepare_contributor_features(data)
    contributor_features = csr_matrix(contributor_features)

    # category features
    category_features = prepare_category_features(data)

    # page type features
    page_type_features = prepare_page_type_features(data)
    page_type_features = csr_matrix(page_type_features)

    # infobox features
    infobox_features = prepare_infobox_features(data)
    infobox_features = csr_matrix(infobox_features).transpose()

    # graph features
    graph_features = prepare_graph_features(data)
    graph_features = csr_matrix(graph_features)

    # join all x features for the algorithms
    x_features = hstack([text_features, link_features, contributor_features, category_features,
                         page_type_features, infobox_features, graph_features])

    # convert to csr matrix (hstack returns a coo sparse matrix)
    # to allow slicing over this matrix when training the algorithms (mini-batching)
    x_features = x_features.tocsr()

    return x_features, y_popularity, y_ranking

