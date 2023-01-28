from calculation import Calculation
from normaziler import Normalizer


def main():
    print("Initializing...")
    normalizer = Normalizer()
    calculation = Calculation()

    while True:
        try:
            query = input("Enter your query please:\t")
            query, query_filters = calculation.get_query_filters_if_any(query)
            query = normalizer.normalize_word_list(query.strip().split())
            ranked_documents = calculation.get_ranked_documents(query, query_filters)
            print(80 * '#')
            for entry in ranked_documents:
                print(f'{entry[0]} ({entry[1]} | {entry[2]}) : {entry[3]}')
            print(80 * '#')
        except Exception:
            continue


main()
