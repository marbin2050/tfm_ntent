__author__ = '{Alfonso Aguado Bustillo}'

import multiprocessing
import json
import pywikibot
import wikitextparser as wtp
import gzip
import pandas as pd
from multiprocessing import Pool


class Parser:

    def __init__(self, parsed_document):
        self.parsed_document = parsed_document

    def get_introduction(self):
        introduction_text = self.parsed_document.sections[0].string
        return introduction_text

    def get_full_text(self, sections):
        full_text = ''
        if sections:
            for section in self.parsed_document.sections:
                if section.title in sections:
                    full_text += section.contents

        return full_text

    def get_section_titles(self):
        return [section.title for section in self.parsed_document.sections if section.title]

    def get_infoboxes(self):
        infoboxes = [temp.string for temp in self.parsed_document.templates if 'infobox' in temp.name.lower()]
        return infoboxes

    def get_cites(self):
        cites = [temp.string for temp in self.parsed_document.templates if 'cite' in temp.name.lower()]
        return cites


# CONTRIBUTORS FEATURES
def get_contributors_features(page):

    # contributors and number of edit per each
    contributors = page.contributors()

    # number of total contributors
    n_contributors = len(contributors)

    # number of total edits
    n_contributors_edits = sum(contributors.values())

    # first edit
    first_edit = str(page.oldest_revision.timestamp)

    # last edit
    last_edit = str(page.latest_revision.timestamp)

    return {'n_contributors': n_contributors,
            'n_contributors_edits': n_contributors_edits,
            'first_edit': first_edit,
            'last_edit': last_edit}


# CATEGORY FEATURES
def get_categories_features(page):

    # categories that the page is in
    categories = []
    for category in page.categories():
        if not category.isEmptyCategory() and not category.isHiddenCategory():
            categories.append(category.title(as_url=True))

    # number of categories
    n_categories = len(categories)

    return {'category_names': categories,
            'n_categories': n_categories}


# LINKS FEATURES
def get_link_features(page):

    # internal links
    internal_links = []
    for internal_link in page.linkedPages():  # omitted interwiki, external, category and image links
        internal_links.append(internal_link.title(as_url=True))

    # pages that link to this page
    back_links = []
    for back_link in page.backlinks():
        back_links.append(back_link.title(as_url=True))

    # number of external links
    n_external_links = sum(1 for x in page.extlinks())

    # number of image links
    n_image_links = sum(1 for x in page.imagelinks())

    # number of language links
    n_lang_links = len(page.langlinks())

    return {'internal_links': internal_links,
            'n_external_links': n_external_links,
            'n_image_links': n_image_links,
            'back_links': back_links,
            'n_lang_links': n_lang_links}


# PAGE TYPE FEATURES
def get_page_type_features(page):

    is_redirect = page.isRedirectPage()
    if is_redirect:
        page = page.getRedirectTarget()

    is_category_page = page.is_categorypage()
    is_category_redirect = page.isCategoryRedirect()
    is_disambig = page.isDisambig()
    is_talkpage = page.isTalkPage()
    is_filepage = page.is_filepage()

    return {'is_redirect': is_redirect,
            'is_category_page': is_category_page,
            'is_category_redirect': is_category_redirect,
            'is_disambig': is_disambig,
            'is_talkpage': is_talkpage,
            'is_filepage': is_filepage}


def get_text_features(page):

    # parser
    parsed_document = wtp.parse(page.text)
    parser = Parser(parsed_document)

    # title
    title = page.title()

    # introduction text
    introduction_text = parser.get_introduction()

    # section titles
    section_titles = parser.get_section_titles()

    # full text
    full_text = parser.get_full_text(section_titles)

    # infoboxes
    infoboxes = parser.get_infoboxes()

    # citations
    citations = parser.get_cites()

    return {'title': title,
            'introduction_text': introduction_text,
            'full_text': full_text,
            'section_titles': section_titles,
            'infoboxes': infoboxes,
            'citations': citations}


def request_page(page_id, views, output_file="data/output_files/100000_top_docs_"):
    try:

        # connection/request
        site = pywikibot.Site('en', 'wikipedia')
        page = pywikibot.Page(site, page_id)

        # if redirect, get the real page
        if page.isRedirectPage():
            page = page.getRedirectTarget()

        # dictionary to store the page data
        data = {}

        # id
        data['page_id'] = page_id

        # url
        data['full_url'] = page.full_url()

        # views / popularity
        data["views"] = views

        # contributors features
        data['contributors'] = get_contributors_features(page)

        # text features
        data['text'] = get_text_features(page)

        # section
        # not included because it is always empty
        # data['section'] = page.section()  # name of the section the Page refers to

        # categories features
        data['categories'] = get_categories_features(page)

        # links features
        data['links'] = get_link_features(page)

        # page type features
        data['page_type'] = get_page_type_features(page)

        # STORE PAGE FEATURES
        json_record = json.dumps(data, ensure_ascii=False)
        file = output_file + str(multiprocessing.current_process().pid) + ".jsonl.gz"
        with gzip.open(file, "ab+") as f:
            f.write((json_record + '\n').encode('utf-8'))

    except Exception as e:
        print("Exception: " + str(e))
        pass


if __name__ == '__main__':

    # REQUEST AND STORE WIKIPEDIA PAGES
    urls_file = "data/input_files/wikipedia/pages_by_views.csv"
    n_top_pages = 300000
    pages = pd.read_csv(urls_file, names=['page', 'views']).sort_values(by='views', ascending=False)
    top_pages = pages[:n_top_pages]
    top_pages = top_pages.reset_index(drop=True)  # reset index

    # RANDOM SAMPLE OF PAGES (TO TRY SMALLER DATASETS)
    top_pages = top_pages.sample(n=100000, random_state=1)
    top_pages = top_pages.sort_values('views', ascending=False)
    top_pages = top_pages.reset_index(drop=True)  # reset index

    # in case some pages have already downloaded
    file = "data/output_files/_100000_top_docs_all_test.gz"
    # load pages already loaded
    with gzip.open(file, "rb") as f:
        for line in f:
            page = json.loads(line)
            page_id = page['page_id']
            top_pages = top_pages[top_pages.page != page_id]

    num_cpus = multiprocessing.cpu_count()
    with Pool(num_cpus) as p:
        p.starmap(request_page, zip(top_pages.iloc[0:n_top_pages - 1, 0], top_pages.iloc[0:n_top_pages - 1, 1]))
