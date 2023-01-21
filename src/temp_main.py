from calculation import Calculation
from normaziler import Normalizer


def main():
    normalizer = Normalizer()
    calculation = Calculation()
    print("Initializing...")
    calculation.make_tables()  # todo : move this to the 'calculation' module

    while True:
        query = input("Enter your query please:\t").strip().split()
        query = normalizer.normalize_word_list(query)
        # query_vector = calculation.make_vector_from_query(query)  # todo : move this to the 'calculation' module
        ranked_documents = calculation.get_ranked_documents(query)
        for entry in ranked_documents.items():
            print(entry[0], ' : ', entry[1])


main()
