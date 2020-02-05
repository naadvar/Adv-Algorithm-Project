Final Project Advanced Algorithm:

This project is to implement a Trie based mini search engine.

## Python Packages required to run this are:
This is written in python 3.6
collections, itertools, string. nlyt, bs4, requests
## How to run the project
Run the python run_this.py file, this will then prompt you to input a search term.
After you enter the term it will go over the weblinks by scapring them and then return the ones that are most relevant.
Right now the top 3 results in decreasing order of relevance are being displayed.


## Algorithm and approach

 - I have used a compressed trie to store and rank words from a few webpages I have parsed. The approach was to make a normal trie using all the data and then compressing it.
 - The way I have implemented the ranking is I count how many times does every word occurs in any page. When an input is being passed, I divide it into single words and then check counts of all of them and combine counts of all the urls, the links which has the most total counts is displayed first.