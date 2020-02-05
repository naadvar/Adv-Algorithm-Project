from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
#this fucntion helps us tokenize the words by removing stopwords, and extracting the text line by line
def tokenizer(sentence):
    swords = stopwords.words('english')
    tokens = []

    for word in word_tokenize(sentence):
        lw = word.lower()

        if word not in punctuation and lw not in swords and word != '':
            tokens.append(lw)

    return tokens
