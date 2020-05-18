from pympler import asizeof
import pandas as pd
import pywikibot
import numpy as np
import json
import jsonlines
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import wikitextparser as wtp
import re
import gzip

# COMMAND LINE ARGUMENTS
wikipedia_data = "input_files/wikipedia/pages_by_views.csv"
top_docs_data = "top_1000_docs.jsonl.gz"
upper_bound = 1000
n_top_pages = 300000


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


def get_doc_data(page_id):
    try:
        # connection/request
        site = pywikibot.Site('en', 'wikipedia')
        page = pywikibot.Page(site, page_id)

        # dictionary to store the document data
        doc = {}

        # id
        doc['page_id'] = page_id
        # full url
        doc['full_url'] = page.full_url()
        # type of content
        doc['is_redirect'] = page.isRedirectPage()
        if doc['is_redirect']:
            page = page.getRedirectTarget()
        doc['is_category_page'] = page.is_categorypage()
        doc['is_category_redirect'] = page.isCategoryRedirect()
        doc['is_disambig'] = page.isDisambig()
        doc['is_talkpage'] = page.isTalkPage()
        doc['is_filepage'] = page.is_filepage()
        doc['is_flowpage'] = page.is_flow_page()

        # get raw content
        doc['title'] = page.title()
        # let's parse the text
        parsed_document = wtp.parse(page.text)
        parser = Parser(parsed_document)
        doc['text'] = parser.extract_contents()

        # get section and categories
        doc['section'] = page.section()
        categories = []
        for category in page.categories():
            categories.append(category.title(as_url=True))
        doc['categories'] = categories

        # get time data
        doc['latest_edit'] = str(page.latest_revision.timestamp)  # latest revision
        doc['oldest_edit'] = str(page.oldest_revision.timestamp)  # first revision

        # get internal links
        links = []
        for link in page.linkedPages():  # interwiki, external, category and image links omitted
            link.title(as_url=True)
        doc['links'] = links
        # get total number of external links
        doc['n_external_links'] = sum(1 for x in page.extlinks())
        # get total number of image links
        doc['n_image_links'] = sum(1 for x in page.imagelinks())

        # store doc
        json_record = json.dumps(doc, ensure_ascii=False)

        with gzip.open(top_docs_data, "ab+") as output_file:
            output_file.write((json_record + '\n').encode('utf-8'))

        # with open(top_docs_data, "a+", encoding='utf-8') as output_file:
        #     output_file.write(json_record + '\n')

    except Exception as e:
        print("Exception: " + str(e))
        pass


if __name__ == '__main__':
    start = time.time()

    # STEP 1: LOAD WIKIPEDIA DATASET
    # read data
    raw_data = pd.read_csv(wikipedia_data, names=['page', 'views'])
    # sort by views
    raw_data = raw_data.sort_values(by='views', ascending=False)
    # get top n pages
    top_pages_data = raw_data[:n_top_pages]
    top_pages_data = top_pages_data.reset_index(drop=True)  # reset index
    print(top_pages_data)

    # STEP 2: GET DATA OF TOP PAGES
    # get document data and store it
    # requests documents in parallel
    with Pool(5) as p:
        p.map(get_doc_data, top_pages_data.iloc[0:upper_bound, 0])

    end = time.time()
    print(end - start)

    # let's check the size iin bytes of each document as well as the text section
    # with jsonlines.open(top_docs_data) as reader:
    #     bytes_list = []
    #     bytes_text_list = []
    #     count_json_files = 0  # count number of documents (lines)
    #     for doc in reader:
    #         count_json_files += 1
    #         # store size in bytes of the document
    #         bytes_list.append(asizeof.asizeof(doc))
    #         # store size in bytes of the text section
    #         bytes_text_list.append(asizeof.asizeof(doc.get('text')))
    #
    # count_lines = list(range(1, count_json_files+1))

    # # ratio between arrays
    # ratio = np.divide(bytes_text_list, bytes_list)

    # # Plot 1
    # plt.scatter(count_lines, bytes_list, alpha=0.2)
    # plt.yscale("log")
    # plt.ylim(0, max(bytes_list))
    # plt.xlim(0, len(count_lines))
    # plt.title('bytes vs num. lines')
    # plt.xlabel('nº lines')
    # plt.ylabel('bytes')
    # plt.show()

    # Plot 2
    # plt.scatter(count_lines, ratio, alpha=0.2)
    # # plt.yscale("log")
    # plt.ylim(0, 1)
    # plt.xlim(0, len(count_lines))
    # plt.title('ratio vs num. lines')
    # plt.xlabel('nº lines')
    # plt.ylabel('ratio')
    # plt.show()
