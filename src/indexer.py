import PyPDF2
import re
import time
import os
import pandas as pd
from PyPDF2 import PageObject
import numpy as np
from normaziler import Normalizer


class Indexer:

    def __init__(self, word_normalizer):
        self.stop_words = self.get_stop_words_from_file('../resources/stop_words_english.txt')
        self.normalizer = word_normalizer

    @staticmethod
    def get_stop_words_from_file(filename: str) -> list:
        with open(filename, 'r', encoding='utf-8') as file:
            stop_words = file.read().splitlines()

            return stop_words

    def main(self):
        start_time = time.time()
        documents_list = self.get_documents_list_and_count()[0]
        final_dict = self.index_all_docs(documents_list)
        self.save_to_file(final_dict)
        print(f'######## it took {time.time() - start_time}s to index all docs ########')

    def index_all_docs(self, documents_list: list):
        final_dict = {}
        for document in documents_list[:1]:
            final_dict = self.index_a_document(document, final_dict)

        return final_dict

    def index_a_document(self, document: str, final_dict: dict) -> dict:
        doc_pages = self.parse_a_document(document)
        for page in doc_pages:
            try:
                page_content = page.extract_text()
                final_dict = self.index_a_page(final_dict, page_content, document)
            except TypeError:
                print('Couldn\'t index the page!')
                pass
            except Exception:
                print('An unknown problem occurred during indexing!')
        return final_dict

    def parse_a_document(self, path_to_doc: str) -> list[PageObject]:
        if not os.getcwd().endswith('repo'):
            os.chdir('../repo')
        reader = PyPDF2.PdfReader(path_to_doc)
        return reader.pages

    def index_a_page(self, final_dict: dict, page_content: str, document: str) -> dict:
        for word in self.get_words_from_text(page_content):
            word = word.lower()

            if not self.should_be_indexed(word):
                continue

            word = self.normalizer.normalize_a_word(word)

            if word not in final_dict.keys():
                final_dict[word] = [1, 1, {}]
                final_dict[word][2] = self.get_including_docs(document, final_dict[word][2])
            else:
                final_dict[word][0] += 1
                final_dict[word][2] = self.get_including_docs(document, final_dict[word][2])
                final_dict[word][1] = len(set(final_dict[word][2]))

        return final_dict

    def get_including_docs(self, doc_name: str, including_docs: dict):
        if including_docs.get(doc_name) is None:
            including_docs[doc_name] = 1
        else:
            including_docs[doc_name] += 1

        return including_docs

    def save_to_file(self, final_dict: dict) -> None:
        df = pd.DataFrame(data=final_dict).T
        df.to_excel('../index.xlsx')

    def get_words_from_text(self, text: str) -> list:
        return re.findall(r'\w+', text)

    def should_be_indexed(self, word: str) -> bool:
        if word in self.stop_words:
            return False
        if word.isdecimal():
            return False
        if word.isdigit():
            return False
        if word.isnumeric():
            return False
        if word.isspace():
            return False
        return True

    def get_term_index_from_list(self, term: str):  # todo 1: specify return type  -  todo2: catch exception
        df = pd.read_excel('../index.xlsx', skiprows=0, usecols='A')
        termlist = np.array(df.values.tolist()).flatten()
        return np.where(termlist == term)[0][0]

    def get_documents_list_and_count(self) -> tuple[list, int]:
        document_list = os.listdir('../repo')
        return document_list, len(document_list)

    def get_term_count(self) -> int:
        df = pd.read_excel('../index.xlsx', skiprows=0, usecols='A')
        return df.shape[0]


if __name__ == '__main__':
    indexer = Indexer()
    indexer.main()
