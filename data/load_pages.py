__author__ = '{Alfonso Aguado Bustillo}'

import gzip
import json
from pympler import asizeof
import pandas as pd
import re


def load_pages(file):

    views = []
    url_length = []
    title_length = []
    text = []
    text_length = []
    text_sections = []
    text_infoboxes = []
    text_cites = []
    text_features = []
    section = []
    categories = []
    latest_edit = []
    oldest_edit = []
    links = []
    external_links = []
    image_links = []
    bytes_list = []
    bytes_text_list = []
    is_redirect = []
    is_category_page = []
    is_category_redirect = []
    is_disambig = []
    is_talkpage = []
    is_filepage = []

    with gzip.open(file, "rb") as f:
        for line in f:
            file = json.loads(line)

            views.append(file["views"])
            url_length.append(len(file["full_url"]))
            title_length.append(len(file["title"]))

            # take a filtered list of words from text
            text.append(re.findall(r'([A-Za-z][a-z]{2,9})', file["text"]["text"]))

            text_length.append(len(file["text"]["text"]))
            if file["text"]["sections"]:
                text_sections.append(len(file["text"]["sections"]))
            else:
                text_sections.append(0)
            if file["text"]["infoboxes"]:
                text_infoboxes.append(len(file["text"]["infoboxes"]))
            else:
                text_infoboxes.append(0)
            if file["text"]["cites"]:
                text_cites.append(len(file["text"]["cites"]))
            else:
                text_cites.append(0)
            if file["text"]["text_features"]:
                text_features.append(len(file["text"]["text_features"]))
            else:
                text_features.append(0)
            if file["section"]:
                section.append(True)
            else:
                section.append(False)
            if file["categories"]:
                categories.append(len(file["categories"]))
            else:
                categories.append(0)
            latest_edit.append(file["latest_edit"])
            oldest_edit.append(file["oldest_edit"])
            if file["links"]:
                links.append(len(file["links"]))
            else:
                links.append(0)
            external_links.append(file["n_external_links"])
            image_links.append(file["n_image_links"])

            bytes_list.append(asizeof.asizeof(file))
            bytes_text_list.append(asizeof.asizeof(file.get('text')))

            is_redirect.append(file["is_redirect"])
            is_category_page.append(file["is_category_page"])
            is_category_redirect.append(file["is_category_redirect"])
            is_disambig.append(file["is_disambig"])
            is_talkpage.append(file["is_talkpage"])
            is_filepage.append(file["is_filepage"])

    # store lists in a dataframe as columns
    data = {'bytes_page': pd.Series(bytes_list),
            'bytes_text': pd.Series(bytes_text_list),
            'views': pd.Series(views),
            "url_length": pd.Series(url_length),
            "title_length": pd.Series(title_length),
            "text": pd.Series(text),
            "text_length": text_length,
            "text_sections": text_sections,
            "text_infoboxes": text_infoboxes,
            "text_cites": text_cites,
            "text_features": text_features,
            "section": section,
            "categories": categories,
            "latest_edit": latest_edit,
            "oldest_edit": oldest_edit,
            "links": links,
            "external_links": external_links,
            "image_links": image_links,
            "is_redirect": is_redirect,
            "is_category_page": is_category_page,
            "is_category_redirect": is_category_page,
            "is_disambig": pd.Series(is_disambig),
            "is_talkpage": pd.Series(is_talkpage),
            "is_filepage": pd.Series(is_filepage)}

    data = pd.DataFrame(data)
    return data
