import argparse

from trie import run



def search(query):
    run(query)


if __name__ == '__main__':
    query = input("Enter the search Term?\n")
    search(query)
