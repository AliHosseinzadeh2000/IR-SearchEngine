import itertools
import math
from operator import itemgetter
import pandas as pd
from src.indexer_v2 import IndexerV2
import numpy as np
import re


class Calculation:

    def __init__(self):
        self.indexer = IndexerV2()
        self.tf_table = None
        self.idf_table = None
        self.tfidf_table = None
        self.docs = None
        self.total_doc_count = None
        self.make_tables()

    def get_ranked_documents(self, query: list[str], query_filters: tuple[str, int] = None):
        query_vector = self.make_vector_from_query(query)
        cosines = dict()

        for index in range(self.total_doc_count):
            try:
                cosines[self.docs[index]] = (self.get_cos_vector(self.get_doc_as_vector(index, self.tfidf_table), query_vector))
            except Exception as e:
                pass

        return self.perform_query_filters(list(dict(sorted(cosines.items(), key=lambda item: item[1], reverse=True)).items())[:10], query_filters)

    def perform_query_filters(self, ranked_documents: list, query_filters: tuple[str, int] = None):
        data_frame = pd.read_excel('../crawled_data.xlsx', skiprows=0, usecols=['link', 'views', 'votes'])
        final_ranked_documents = []

        for item in ranked_documents:
            link = 'https://stackoverflow.com' + item[0]
            score = item[1]
            row = np.where(data_frame.values == item[0])[0][0]
            views = data_frame['views'][row]
            votes = data_frame['votes'][row]
            final_ranked_documents.append([link, views, votes, score])

        if query_filters:
            temp_ranked_documents = final_ranked_documents
            final_ranked_documents = []
            if query_filters[0] == 'views':
                for item in temp_ranked_documents:
                    if item[1] >= query_filters[1]:
                        final_ranked_documents.append(item)
                return sorted(final_ranked_documents, key=itemgetter(1))
            elif query_filters[0] == 'votes':
                for item in temp_ranked_documents:
                    if item[2] >= query_filters[1]:
                        final_ranked_documents.append(item)
                return sorted(final_ranked_documents, key=itemgetter(2))
        else:
            return final_ranked_documents

    def get_query_filters_if_any(self, query: str) -> tuple[str, tuple[str, int]] | tuple[str, None]:
        filters_pattern = '(VIEWS:[0-9]+)|(VOTES:[0-9]+)'
        match_obj = re.search(filters_pattern, query)
        if match_obj:
            filter_name = re.findall('(.*?):', match_obj[0])[0].lower()
            filter_number = int(re.findall('\d+', match_obj[0])[0])

            query = re.sub(filters_pattern, '', query)
            return query, (filter_name, filter_number)
        else:
            return query, None

    def make_tables(self) -> None:
        # making index table
        try:
            if self.indexer.index_file.empty:
                print("Indexing documents...")
                self.indexer.main()
        except Exception:
            print("Indexing documents...")
            self.indexer.main()

        self.docs, self.total_doc_count = self.indexer.get_indexed_documents_list_and_count()
        self.docs = sorted(self.docs, key=str.casefold)

        try:
            if not self.tf_table.empty:
                pass
        except AttributeError:
            try:
                self.tf_table = pd.read_excel('../tf_table.xlsx', skiprows=0)
                if self.tf_table.empty:
                    print("Extracting tf table...")
                    self.extract_tf_table()
            except Exception:
                print("Extracting tf table...")
                self.extract_tf_table()
        except Exception:
            print("Extracting tf table...")
            self.extract_tf_table()

        try:
            if not self.idf_table.empty:
                pass
        except AttributeError:
            try:
                self.idf_table = pd.read_excel('../idf_table.xlsx', skiprows=0, usecols='B')
                if self.idf_table.empty:
                    print("Calculating idf...")
                    self.extract_idf_table(self.total_doc_count)
            except Exception:
                print("Calculating idf...")
                self.extract_idf_table(self.total_doc_count)
        except Exception:
            print("Calculating idf...")
            self.extract_idf_table(self.total_doc_count)

        try:
            if not self.tfidf_table.empty:
                pass
        except AttributeError:
            try:
                self.tfidf_table = pd.read_excel('../tfidf.xlsx', skiprows=0)
                if self.tfidf_table.empty:
                    print("Calculating tf-idf...")
                    self.extract_tfidf_table()
            except Exception:
                print("Calculating tf-idf...")
                self.extract_tfidf_table()
        except Exception:
            print("Calculating tf-idf...")
            self.extract_tfidf_table()

    def dot_product(self, vector1: list[int | float], vector2: list[int | float]) -> float:
        new_vector = []

        if len(vector1) == len(vector2):
            new_vector.append([vec1 * vec2 for vec1, vec2 in zip(vector1, vector2)])
            total_sum = np.sum(np.array(new_vector))
            return total_sum
        else:
            print("Vectors are not the same size")

    def calculate_normal_idf(self, df: int, total_doc_count: int) -> float:
        return math.log(total_doc_count / df, 10)

    def calculate_normal_idf_for_all(self, dfs: list[int], total_doc_count: int) -> list[float]:
        idfs = [math.log(total_doc_count / df, 10) for df in dfs]
        return idfs

    def calculate_normal_tfidf(self, tf: int, idf: float) -> float:
        return float(math.log(1 + float(tf), 10)) * idf

    # Calculates tf-idf for a complete row (for a term in each document)
    def calculate_normal_tfidf_for_row(self, tfs: list[int], idf: float) -> list[float]:
        tfidf = [math.log(1 + tf, 10) * idf for tf in tfs]
        return tfidf

    # Calculates tf-idf for a complete column
    def calculate_normal_tfidf_for_col(self, tfs: list[int], idfs: list[float]) -> list[float]:
        tfidf = [math.log(1 + tf, 10) * idf for tf, idf in zip(tfs, idfs)]
        return tfidf

    def get_vector_size(self, vector: list[int | float]) -> float:
        powered_list = [math.pow(vec, 2) for vec in vector]
        total_sum = np.sum(np.array(powered_list))
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
        data_frame = self.indexer.index_file[:-2].loc[:, ['Unnamed: 0', 'including_docs']]

        terms = []
        docs = set()
        datas = []

        column = data_frame.iloc[:, 0]
        terms.append(list(column))
        terms = list(itertools.chain(*terms))

        for index in range(len(data_frame.values)):
            pair = dict(eval(data_frame.values[index][1]))
            docs.update(pair.keys())

        docs_list = list(docs)
        docs_list = sorted(docs_list, key=str.casefold)

        for index in range(len(data_frame.values)):
            doc_tf_dict = dict(eval(data_frame.values[index][1]))
            data = [0] * len(docs)
            for index2 in range(len(doc_tf_dict)):
                data[docs_list.index(list(doc_tf_dict.keys())[index2])] = list(doc_tf_dict.values())[index2]
            datas.append(data)

        new_df = pd.DataFrame(datas, index=terms, columns=docs_list)
        new_df.to_excel('../tf_table.xlsx')
        self.tf_table = pd.read_excel('../tf_table.xlsx', skiprows=0)

    def extract_idf_table(self, total_doc_count: int) -> None:
        data_frame = self.indexer.index_file[:-2].loc[:, ['Unnamed: 0', 'df']]
        terms = []
        dfs = []

        column = data_frame.iloc[:, 0]
        terms.append(list(column))
        terms = list(itertools.chain(*terms))

        column = data_frame.iloc[:, 1]
        dfs.append(list(column))
        dfs = list(itertools.chain(*dfs))

        new_df = pd.DataFrame(self.calculate_normal_idf_for_all(dfs, total_doc_count), index=terms)
        new_df.to_excel('../idf_table.xlsx')
        self.idf_table = pd.read_excel('../idf_table.xlsx', skiprows=0, usecols='B')

    def extract_tfidf_table(self):
        idfs = self.idf_table.values.tolist()
        idfs = list(itertools.chain(*idfs))
        cols = self.tf_table[self.tf_table.columns[1:]]

        tfidf = []
        for index in range(self.total_doc_count):
            tfidf.append(self.calculate_normal_tfidf_for_col(list(cols.iloc[:, index]), idfs))

        new_data_frame = pd.DataFrame(tfidf)
        new_data_frame = new_data_frame.transpose()
        new_data_frame.to_excel('../tfidf.xlsx')
        self.tfidf_table = pd.read_excel('../tfidf.xlsx', skiprows=0)

    def get_doc_as_vector(self, doc_num: int, data_frame: pd.DataFrame) -> list[float]:
        all_cols = data_frame[data_frame.columns[1:]]
        mylist = all_cols.iloc[:, doc_num]
        return mylist
