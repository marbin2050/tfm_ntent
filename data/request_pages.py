__author__ = '{Alfonso Aguado Bustillo}'

import multiprocessing
import json
import pywikibot
import wikitextparser as wtp
import re
import gzip
import pandas as pd
from multiprocessing import Pool


class Parser:

    def __init__(self, parsed_document):
        self.parsed_document = parsed_document

    def get_introduction(self):
        introduction_text = self.parsed_document.sections[0].string
        return re.sub('\{\{.*\}\}\n', '', introduction_text, flags=re.S)

    def get_titles(self):
        return [sec.title for sec in self.parsed_document.sections if sec.title]

    def get_infoboxes(self):
        infoboxes = [temp.string for temp in self.parsed_document.templates if 'infobox' in temp.name.lower()]
        return infoboxes

    def get_cites(self):
        cites = [temp.string for temp in self.parsed_document.templates if 'cite' in temp.name.lower()]
        return cites

    def compute_features(self):
        return None

    def extract_contents(self):
        introduction_text = self.get_introduction()
        section_titles = self.get_titles()
        infoboxes = self.get_infoboxes()
        citations = self.get_cites()
        features = self.compute_features()
        return {'text': introduction_text,
                'sections': section_titles,
                'infoboxes': infoboxes,
                'cites': citations,
                'text_features': features}


def get_page(page_id, views, output_file="output_files/top_docs_"):
    try:

        # connection/request
        site = pywikibot.Site('en', 'wikipedia')
        page = pywikibot.Page(site, page_id)

        # dictionary to store the document data
        data = {}
        data['page_id'] = page_id
        data["views"] = views

        data['full_url'] = page.full_url()
        data['title'] = page.title()

        # let's parse the text
        parsed_document = wtp.parse(page.text)
        parser = Parser(parsed_document)
        data['text'] = parser.extract_contents()

        data['section'] = page.section()
        categories = []
        for category in page.categories():
            categories.append(category.title(as_url=True))
        data['categories'] = categories

        data['latest_edit'] = str(page.latest_revision.timestamp)  # latest revision
        data['oldest_edit'] = str(page.oldest_revision.timestamp)  # first revision

        links = []
        for link in page.linkedPages():  # interwiki, external, category and image links omitted
            links.append(link.title(as_url=True))
        data['links'] = links
        data['n_external_links'] = sum(1 for x in page.extlinks())  # number of external links
        data['n_image_links'] = sum(1 for x in page.imagelinks())  # number of image links

        # type of page
        data['is_redirect'] = page.isRedirectPage()
        if data['is_redirect']:
            page = page.getRedirectTarget()
        data['is_category_page'] = page.is_categorypage()
        data['is_category_redirect'] = page.isCategoryRedirect()
        data['is_disambig'] = page.isDisambig()
        data['is_talkpage'] = page.isTalkPage()
        data['is_filepage'] = page.is_filepage()

        # store page
        json_record = json.dumps(data, ensure_ascii=False)
        file = output_file + str(multiprocessing.current_process().pid) + ".jsonl.gz"
        with gzip.open(file, "ab+") as f:
            f.write((json_record + '\n').encode('utf-8'))

    except Exception as e:
        print("Exception: " + str(e))
        pass


if __name__ == '__main__':

    # REQUEST AND STORE WIKIPEDIA PAGES
    urls_file = "input_files/wikipedia/pages_by_views.csv"
    n_top_pages = 300000
    pages = pd.read_csv(urls_file, names=['page', 'views']).sort_values(by='views', ascending=False)
    top_pages = pages[:n_top_pages]
    top_pages = top_pages.reset_index(drop=True)  # reset index

    num_cpus = multiprocessing.cpu_count()
    with Pool(num_cpus) as p:
        p.starmap(get_page, zip(top_pages.iloc[0:n_top_pages - 1, 0], top_pages.iloc[0:n_top_pages - 1, 1]))
