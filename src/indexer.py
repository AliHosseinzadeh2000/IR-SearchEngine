import PyPDF2
import re
import time
import os
import pandas as pd
from PyPDF2 import PageObject


class Indexer:

    def __init__(self):
        self.stop_words = self.get_stop_words_from_file('../resources/stop_words_english.txt')

    def main(self):
        start_time = time.time()
        documents_list = os.listdir('../repo/')
        final_dict = self.index_all_docs(documents_list)
        self.save_to_file(final_dict)
        print(f'######## it took {time.time() - start_time}s to finish ########')

    def index_all_docs(self, documents_list: list):
        final_dict = {}
        for document in documents_list[:3]:
            final_dict = self.index_a_document(document, final_dict)

        return final_dict

    @staticmethod
    def get_stop_words_from_file(filename: str) -> list:
        with open(filename, 'r', encoding='utf-8') as file:
            stop_words = file.read().splitlines()

            return stop_words

    def index_a_document(self, document: str, final_dict: dict) -> dict:
        doc_pages = self.parse_a_document(document)
        for page in doc_pages:
            try:
                page_content = page.extract_text()
                final_dict = self.index_a_page(final_dict, page_content, document)
            except TypeError:
                print('couldn\'t index the page!')
                pass
            except Exception:
                print('an unknown problem occurred during indexing!')
        return final_dict

    def parse_a_document(self, path_to_doc: str) -> list[PageObject]:
        if not os.getcwd().endswith('repo'):
            os.chdir('../repo')
        reader = PyPDF2.PdfReader(path_to_doc)
        return reader.pages

    def index_a_page(self, final_dict: dict, page_content: str, document: str) -> dict:
        for word in self.get_words_from_text(page_content):
            if word.lower() in self.stop_words:
                continue

            if not word.lower() in final_dict.keys():  # todo : add more conditions & checks
                final_dict[word.lower()] = [1, [document]]
            else:
                final_dict[word.lower()][0] += 1
                final_dict[word.lower()][1].append(document)

        return final_dict

    def save_to_file(self, final_dict: dict) -> None:
        df = pd.DataFrame(data=final_dict).T
        df.to_excel('index.xlsx')

    def get_words_from_text(self, text: str) -> list:
        return re.findall(r'\w+', text)


if __name__ == '__main__':
    indexer = Indexer()
    indexer.main()
