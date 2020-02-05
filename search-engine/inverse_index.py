from collections import defaultdict
from tokenizer import tokenizer
#here we add the data to the inverse index by tokenizing the words and then adding it to the dictionary
def inverse_index(data):
    d_map = defaultdict(list)

    for idx, val in enumerate(data):
        for word in tokenizer(val):
            d_map[word].append(idx)

    return d_map