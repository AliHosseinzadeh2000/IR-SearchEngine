from calculation import Calculation
from normaziler import Normalizer


def main():
    normalizer = Normalizer()
    calculation = Calculation()
    print("Initializing...")

    while True:
        query = input("Enter your query please:\t").strip().split()
        query = normalizer.normalize_word_list(query)
        ranked_documents = calculation.get_ranked_documents(query)
        for entry in ranked_documents.items():
            print(f'{entry[0]} : {entry[1]}')


main()
