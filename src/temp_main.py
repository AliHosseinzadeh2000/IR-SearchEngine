from indexer import Indexer
import calculation
from normaziler import Normalizer
import input


def main():
    normalizer = Normalizer()
    print("Initializing...\nIndexing documents...\n")
    indexer = Indexer(normalizer)
    indexer.main()
    while True:
        query = input.init_input()
        query = normalize_word_list(query, normalizer)
        print(query)


def normalize_word_list(word_list, normalizer):
    for index in range(len(word_list)):
        word_list[index] = normalizer.normalize_a_word(word_list[index])
    return word_list

main()
