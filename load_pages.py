__author__ = '{Alfonso Aguado Bustillo}'

from datetime import datetime
import gzip
import json
from pympler import asizeof
import pandas as pd
import re


class FeatureEngineering:

    def __init__(self, page, data):
        self.page = page
        self.data = data

    def load_text_features(self):

        # title
        title = self.data.get('title')
        title.append(self.page['text']['title'])

        # introduction text
        introduction_text = self.data.get('introduction_text')
        introduction_text.append(self.page['text']['introduction_text'])

        # full text
        # full_text = self.data.get('full_text')
        # full_text.append(self.page['text']['full_text'])

        # title length in characters
        title = self.page['text']['title']
        title = re.sub(r'\W+', ' ', title)  # remove non-alphanumeric characters via regex
        title_length = self.data.get('title_length')
        title_length.append(len(title))

        # title length in words
        n_title_words = self.data.get('n_title_words')
        n_title_words.append(len(title.split()))

        # introduction length in words
        n_words_introduction = self.data.get('n_introduction_words')
        n_words_introduction.append(len(self.page['text']['introduction_text']))

        # full text length in words
        n_words_full_text = self.data.get('n_full_text_words')
        n_words_full_text.append(len(self.page['text']['full_text']))

        # size introduction in bytes
        bytes_introduction_text = self.data.get('bytes_introduction_text')
        bytes_introduction_text.append(asizeof.asizeof(self.page['text']['introduction_text']))

        # size full text in bytes
        bytes_full_text = self.data.get('bytes_full_text')
        bytes_full_text.append(asizeof.asizeof(self.page['text']['full_text']))

        # number of citations
        n_citations = self.data.get('n_citations')
        if len(self.page['text']['citations']) != 0:
            n_citations.append(len(self.page['text']['citations']))
        else:
            n_citations.append(0)

        # number of sections
        n_sections = self.data.get('n_sections')
        if len(self.page['text']['section_titles']) != 0:
            n_sections.append(len(self.page['text']['section_titles']))
        else:
            n_sections.append(0)

        # infoboxes
        infoboxes = self.data.get('infoboxes')
        infoboxes.append(self.page['text']['infoboxes'])

    def load_contributor_features(self):

        # number of contributors
        n_contributors = self.data.get('n_contributors')
        n_contributors.append(self.page['contributors']['n_contributors'])

        # number of edits done by contributors
        n_contributors_edits = self.data.get('n_contributors_edits')
        n_contributors_edits.append(self.page['contributors']['n_contributors_edits'])

        # last edit since today
        time_since_last_edit = self.data.get('time_since_last_edit')
        last_edit = self.page['contributors']['last_edit']
        last = datetime.timestamp(datetime.strptime(last_edit, "%Y-%m-%dT%H:%M:%SZ"))
        now = datetime.timestamp(datetime.now())
        scaled_time = (now-last) / (60 * 60 * 24)
        time_since_last_edit.append(scaled_time)
        # time_since_last_edit.append(now-last)

        # first edit since today
        time_since_first_edit = self.data.get('time_since_first_edit')
        first_edit = self.page['contributors']['first_edit']
        first = datetime.timestamp(datetime.strptime(first_edit, "%Y-%m-%dT%H:%M:%SZ"))
        scaled_time = (now - first) / (60 * 60 * 24)
        time_since_first_edit.append(scaled_time)
        # time_since_first_edit.append(now - first)

    def load_category_features(self):

        # categories
        categories = self.data.get('category_names')
        categories.append(self.page['categories']['category_names'])

        # number of categories
        n_categories = self.data.get('n_categories')
        n_categories.append(self.page['categories']['n_categories'])

    def load_link_features(self):

        # page url
        full_url = self.data.get('full_url')
        full_url.append(self.page['full_url'])

        # internal links
        internal_links = self.data.get('internal_links')
        internal_links.append(self.page['links']['internal_links'])

        # number of internal links
        n_internal_links = self.data.get('n_internal_links')
        n_internal_links.append(len(self.page['links']['internal_links']))

        # number of external links
        n_external_links = self.data.get('n_external_links')
        n_external_links.append(self.page['links']['n_external_links'])

        # number of image links
        n_image_links = self.data.get('n_image_links')
        n_image_links.append(self.page['links']['n_image_links'])

        # back links
        back_links = self.data.get('back_links')
        back_links.append(self.page['links']['back_links'])

        # number of back links
        n_back_links = self.data.get('n_back_links')
        n_back_links.append(len(self.page['links']['back_links']))

        # number of language links
        n_lang_links = self.data.get('n_lang_links')
        n_lang_links.append(self.page['links']['n_lang_links'])

    def load_page_type_features(self):

        # is redirect page
        # is_redirect = self.data.get('is_redirect')
        # is_redirect.append(self.page['page_type']['is_redirect'])

        # is category page
        is_category_page = self.data.get('is_category_page')
        is_category_page.append(self.page['page_type']['is_category_page'])

        # is_category_redirect page
        # is_category_redirect = self.data.get('is_category_redirect')
        # is_category_redirect.append(self.page['page_type']['is_category_redirect'])

        # is_disambig page
        is_disambig = self.data.get('is_disambig')
        is_disambig.append(self.page['page_type']['is_disambig'])

        # is_talkpage page
        is_talkpage = self.data.get('is_talkpage')
        is_talkpage.append(self.page['page_type']['is_talkpage'])

        # is_filepage page
        is_filepage = self.data.get('is_filepage')
        is_filepage.append(self.page['page_type']['is_filepage'])


def load_pages(file):

    # dictionary to store the pages data
    data = {'views': [],
            'title': [],
            'title_length': [],
            'introduction_text': [],
            # 'full_text': [],
            'n_title_words': [],
            'n_introduction_words': [],
            'n_full_text_words': [],
            'bytes_introduction_text': [],
            'bytes_full_text': [],
            'n_citations': [],
            'n_sections': [],
            'infoboxes': [],
            'full_url': [],
            'internal_links': [],
            'n_internal_links': [],
            'n_external_links': [],
            'n_image_links': [],
            'back_links': [],
            'n_back_links': [],
            'n_lang_links': [],
            'n_contributors': [],
            'n_contributors_edits': [],
            'time_since_last_edit': [],
            'time_since_first_edit': [],
            'category_names': [],
            'n_categories': [],
            # 'is_redirect': [],
            'is_category_page': [],
            # 'is_category_redirect': [],
            'is_disambig': [],
            'is_talkpage': [],
            'is_filepage': []}

    with gzip.open(file, "rb") as f:
        for line in f:
            page = json.loads(line)

            feature_eng = FeatureEngineering(page, data)

            # views/popularity
            views = data.get('views')
            views.append(page['views'])

            # load text features
            feature_eng.load_text_features()

            # load contributor features
            feature_eng.load_contributor_features()

            # load link features
            feature_eng.load_link_features()

            # load category features
            feature_eng.load_category_features()

            # page type features
            feature_eng.load_page_type_features()

    data = pd.DataFrame(data)
    return data
