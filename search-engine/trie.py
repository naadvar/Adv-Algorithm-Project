from collections import defaultdict, Counter
from itertools import chain
from node import Node
from node import add,find,extract_prefix,ranking
from string import punctuation

from nltk import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from scraper import clean_html
from tokenizer import tokenizer
from inverse_index import inverse_index
import requests


MAIN_MAP = defaultdict(list)





def hit_urls(url_list):

    return [clean_html(requests.get(url).text) for url in url_list]


# tokenize doc - removes stopwords and punctuations


# map of word to doc occurrence



# compressed trie nodes



def run(sq):
    url_list =[
        "https://isha.sadhguru.org/us/en/wisdom/article/what-to-eat-making-right-food-choices",
        "https://www.pythonforbeginners.com/basics/getting-user-input-from-the-keyboard",
        "https://medium.com/center-for-data-science/deepmind-fellow-profile-ksenia-saenko-e6d0f7574a59",
        "https://medium.com/center-for-data-science/deepmind-fellow-profile-yassine-kadiri-7bfe4a045050"
        ]
    data = inverse_index(hit_urls(url_list))

    # update main map with words from the html pages, with their occurrences
    MAIN_MAP.update(data)

    query = tokenizer(sq)

    root = Node()
    ignore = ['©', '—', '’', '“', '”', "''"]

    for word in MAIN_MAP:
        if word not in ignore:
            add(root, word)

    retval = {}

    # search the compressed trie using the find function
    for key in query:
        if find(root, key):
            retval.update({key: MAIN_MAP[key]})

    resulting_idx = ranking(retval)

    if not resulting_idx:
        print(f'\n No results for your search query - {sq}')
        print('\n  Modify the query and try again, listed below are the searched URLs')

        for idx, ul in enumerate(url_list):
            print(f'{idx+1}.{ul}')

        return

    print("\n Search results, in decreasing order of relevance \n")
    for idx, val in enumerate(resulting_idx):
        print(f'{idx+1}: {url_list[val]}')
