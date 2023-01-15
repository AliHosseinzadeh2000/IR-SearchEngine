from indexer import Indexer
import calculation
from normaziler import Normalizer
import input


def main():
    normalizer = Normalizer()
    indexer = Indexer(normalizer)
    print("Initializing...")
    make_tables(indexer)

    while True:
        query = input.init_input()
        query = normalize_word_list(query, normalizer)
        query_vector = calculation.make_vector_from_query(indexer, query)


def normalize_word_list(word_list, normalizer):
    for index in range(len(word_list)):
        word_list[index] = normalizer.normalize_a_word(word_list[index])
    return word_list


def make_tables(indexer):
    # making index table
    print("Indexing documents...")
    indexer.main()

    print("Extracting tf table...")
    calculation.extract_tf_table()

    print("Calculating idf...")
    # 5 'cause current indexed document count is 5
    calculation.extract_idf_table(5)

    print("Calculating tf-idf...")

main()
