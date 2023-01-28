from calculation import Calculation
from normaziler import Normalizer


def main():
    print("Initializing...")
    normalizer = Normalizer()
    calculation = Calculation()

    while True:
        query = input("Enter your query please:\t").strip().split()
        query = normalizer.normalize_word_list(query)
        ranked_documents = calculation.get_ranked_documents(query)
        for entry in list(ranked_documents.items())[:10]:
            print(entry[0], ' : ', entry[1])


main()
