import math
import pandas as pd
from src.indexer import Indexer


class Calculation:

    def __init__(self):
        self.indexer = Indexer()

    def get_ranked_documents(self, query: list[str]):
        query_vector = self.make_vector_from_query(query)
        cosines = []
        data_frame = pd.read_excel('../tfidf.xlsx', skiprows=0)
        for index in range(5):
            cosines.append(self.get_cos_vector(self.get_doc_as_vector(index, data_frame), query_vector))
        return cosines

    def make_tables(self) -> None:
        # making index table
        try:
            data_frame = pd.read_excel('../index.xlsx')
            if data_frame.empty:
                print("Indexing documents...")
                self.indexer.main()
        except Exception:
            print("Indexing documents...")
            self.indexer.main()

        print("Extracting tf table...")
        self.extract_tf_table()

        print("Calculating idf...")
        # total_doc_count = indexer.get_documents_list_and_count()[1]
        self.extract_idf_table(5)  # 5 -> cause current indexed document count is 5

        print("Calculating tf-idf...")
        self.extract_tfidf_table()

    def dot_product(self, vector1: list[int | float], vector2: list[int | float]) -> float:
        new_vector = []
        total_sum = 0

        if len(vector1) == len(vector2):
            for index in range(len(vector1)):
                new_vector.append(vector1[index] * vector2[index])
            for index in range(len(new_vector)):
                total_sum += new_vector[index]
            return total_sum
        else:
            print("Vectors are not the same size")

    def calculate_normal_idf(self, df: int, total_doc_count: int) -> float:
        return math.log(total_doc_count / df, 10)

    def calculate_normal_idf_for_all(self, dfs: list[int], total_doc_count: int) -> list[float]:
        idfs = []
        for index in range(len(dfs)):
            idfs.append(self.calculate_normal_idf(dfs[index], total_doc_count))
        return idfs

    def calculate_normal_tfidf(self, tf: int, idf: float) -> float:
        return float(math.log(1 + float(tf), 10)) * idf

# Calculates tf-idf for a complete row (for a term in each document)
    def calculate_normal_tfidf_for_all(self, tfs: list[int], idf: float) -> list[float]:
        tfidf = []
        for index in range(len(tfs)):
            tfidf.append(self.calculate_normal_tfidf(tfs[index], idf))
        return tfidf

    def get_vector_size(self, vector: list[int | float]) -> float:
        total_sum = 0
        for index in range(len(vector)):
            total_sum += math.pow(vector[index], 2)
        return math.sqrt(total_sum)

    def get_cos_vector(self, vector1: list[int | float], vector2: list[int | float]) -> float:
        dot_result = self.dot_product(vector1, vector2)
        vector1_size = self.get_vector_size(vector1)
        vector2_size = self.get_vector_size(vector2)
        return dot_result / (vector1_size * vector2_size)

    def make_vector_from_query(self, query_list: list[str]) -> list[int]:
        vector = [0] * self.indexer.get_term_count()
        for word in query_list:
            try:
                vector[self.indexer.get_term_index_from_list(word)] += 1
            except Exception:
                pass
        return vector

    def extract_tf_table(self) -> None:
        data_frame = pd.read_excel('../index.xlsx', skiprows=0, usecols='A, D')
        terms = []

        for index in range(len(data_frame.values)):
            terms.append(data_frame.values[index][0])

        docs = set()
        datas = []

        for index2 in range(len(data_frame.values)):
            pair = dict(eval(data_frame.values[index2][1]))
            docs.update(pair.keys())

        docs_list = list(docs)

        for index3 in range(len(data_frame.values)):
            doc_tf_dict = dict(eval(data_frame.values[index3][1]))
            data = [0] * len(docs_list)
            for index4 in range(len(doc_tf_dict)):
                data[docs_list.index(list(doc_tf_dict.keys())[index4])] = list(doc_tf_dict.values())[index4]
            datas.append(data)

        new_df = pd.DataFrame(datas, index=terms, columns=docs_list)
        new_df.to_excel('../tf_table.xlsx')

    def extract_idf_table(self, total_doc_count: int) -> None:
        data_frame = pd.read_excel('../index.xlsx', skiprows=0, usecols='A, C')
        terms = []
        for index in range(len(data_frame.values)):
            terms.append(data_frame.values[index][0])

        dfs = []
        for index in range(len(data_frame.values)):
            dfs.append(data_frame.values[index][1])

        new_df = pd.DataFrame(self.calculate_normal_idf_for_all(dfs, total_doc_count), index=terms)
        new_df.to_excel('../idf_table.xlsx')


    def extract_tfidf_table(self):
        data_frame = pd.read_excel('../idf_table.xlsx', skiprows=0, usecols='B')
        idfs = data_frame.values.tolist()
        flat_idfs = sum(idfs, [])

        data_frame = pd.read_excel('../tf_table.xlsx', skiprows=0)
        cols = data_frame[data_frame.columns[1:]]
        index = 0
        tfidf = []
        for col in cols.values:
            tfidf.append(self.calculate_normal_tfidf_for_all(col, flat_idfs[index]))
            index += 1

        # TODO: Put the doc names on top of the Excel file for user presentation - columns=docs_list
        new_df = pd.DataFrame(tfidf)
        new_df.to_excel('../tfidf.xlsx')

    def get_doc_as_vector(self, doc_num: int, data_frame: pd.DataFrame) -> list[float]:
        # TODO: Keep Excel file in memory for optimization
        all_cols = data_frame[data_frame.columns[1:]]
        return [i[doc_num] for i in all_cols.values]
