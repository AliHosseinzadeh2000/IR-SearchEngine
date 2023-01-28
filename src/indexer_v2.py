from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import numpy as np
import re
import schedule
from src.normaziler import Normalizer
import ast


class IndexerV2:

    def __init__(self):
        self.try_to_read_index_file()
        self.indexed_documents_list = None
        self.indexed_documents_count = None
        self.stop_words = self.get_stop_words_from_file('../resources/stop_words_english.txt')
        self.normalizer = Normalizer()

    def try_to_read_index_file(self) -> None:
        try:
            self.index_file = pd.read_excel('../index.xlsx', skiprows=0)
        except Exception:
            self.index_file = None
            pass

    @staticmethod
    def get_stop_words_from_file(filename: str) -> list[str]:
        with open(filename, 'r', encoding='utf-8') as file:
            stop_words = file.read().splitlines()

            return stop_words

    def run_scheduler(self):
        print('The scheduler will be triggered in a second...')
        schedule.every(1).minutes.do(self.main)

        while True:
            schedule.run_pending()
            time.sleep(1)

    def main(self) -> None:
        start_time = time.time()
        list_of_questions = self.crawl_list_of_questions()
        self.crawl_detail_info(list_of_questions)
        final_dict = self.index_all_docs()
        self.save_to_file(final_dict)
        print(f'~~~~~~~~~~ it took {time.time() - start_time}s to index all docs ~~~~~~~~~~')
        self.index_file = pd.read_excel('../index.xlsx', skiprows=0)
        self.indexed_documents_list, self.indexed_documents_count = self.get_indexed_documents_list_and_count()

    def crawl_list_of_questions(self) -> list:
        url = 'https://stackoverflow.com/questions?tab=votes&pagesize=50'
        all_questions = []

        print(f'The spider has started crawling...\U0001F577\U0001F578')

        page_number = 1
        for _ in range(1, 3):  # specify the number of stackoverflow pages to crawl
            if page_number != 1:
                url = f'https://stackoverflow.com/questions?tab=votes&page={page_number}'

            soup = self._make_soup_obj(url)

            questions_section = soup.find('div', class_='flush-left')

            questions = questions_section.find_all(class_='s-post-summary js-post-summary')

            for num, question in enumerate(questions):
                title = question.find('h3', class_='s-post-summary--content-title').text.strip()
                link = question.find('h3', class_='s-post-summary--content-title').a.get('href')  # https://stackoverflow.com + ...

                all_questions.append([title, link])

            print(f'\textracting links from page {page_number}: Done')
            page_number += 1

        return all_questions

    def crawl_detail_info(self, list_of_questions: list[list[str]]) -> None:
        final_list = []
        for question in list_of_questions:
            try:
                title = question[0]
                link = question[1]
                content, views, votes = self.add_page_content_and_views_and_votes_to_excel('https://stackoverflow.com' + link)

                final_list.append([title, link, content, views, votes])
            except Exception:
                pass

        df = pd.DataFrame(final_list, columns=['title', 'link', 'content', 'views', 'votes'])
        df.to_excel('../crawled_data.xlsx')

    def add_page_content_and_views_and_votes_to_excel(self, url: str) -> tuple[str, int, int]:
        soup = self._make_soup_obj(url)

        page_content = soup.find('div', id='mainbar').get_text()
        page_content = re.sub('[\n|\r]', '', page_content)

        views = int(''.join(filter(str.isdigit, soup.find('div', class_='flex--item ws-nowrap mb8')['title'].strip())))
        votes = int(soup.find('div', class_='js-vote-count flex--item d-flex fd-column ai-center fc-black-500 fs-title').text.strip())

        return page_content, views, votes

    def _make_soup_obj(self, url: str) -> BeautifulSoup:
        page = ''
        while page == '':
            try:
                page = requests.get(url=url)
                break
            except:
                print('Connection refused by the server..')
                print('Let me sleep for 5 seconds')
                print('ZZzzzz...')
                time.sleep(1)
                print('Was a nice sleep, now let me continue...')
                continue

        return BeautifulSoup(page.content, 'html.parser')

    def index_all_docs(self) -> [dict]:
        terms_dict = {}
        df = pd.read_excel('../crawled_data.xlsx', skiprows=0, usecols=['title', 'link', 'content', 'views', 'votes'])
        indexed_documents_count = 0
        indexed_documents_list = []
        for item in df.values.tolist():
            try:
                self.index_a_document(item[2], item[1], terms_dict)
                indexed_documents_count += 1
                indexed_documents_list.append(item[1])
            except Exception:
                pass

        terms_dict['indexed_documents_count'] = [indexed_documents_count, 0, 0]
        terms_dict['indexed_documents_list'] = [indexed_documents_list, 0, 0]
        return terms_dict

    def index_a_document(self, document_text: str, document_link: str, terms_dict: dict) -> dict:
        for word in self.get_words_from_text(document_text):
            try:
                word = word.lower()

                if not self.should_be_indexed(word):
                    continue

                word = self.normalizer.normalize_a_word(word)

                if word not in terms_dict.keys():
                    terms_dict[word] = [1, 1, {}]
                    terms_dict[word][2] = self.get_including_docs(document_link, terms_dict[word][2])
                else:
                    terms_dict[word][0] += 1
                    terms_dict[word][2] = self.get_including_docs(document_link, terms_dict[word][2])
                    terms_dict[word][1] = len(set(terms_dict[word][2]))
            except Exception:
                print('An unknown problem occurred during indexing!')
        return terms_dict

    def get_including_docs(self, doc_name: str, including_docs: dict):
        if including_docs.get(doc_name) is None:
            including_docs[doc_name] = 1
        else:
            including_docs[doc_name] += 1

        return including_docs

    def save_to_file(self, final_dict: dict) -> None:
        df = pd.DataFrame(data=final_dict).transpose()
        df.columns = ['cf', 'df', 'including_docs']
        df.to_excel('../index.xlsx')

    def get_words_from_text(self, text: str) -> list[str]:
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

    def get_documents_list(self) -> tuple[list, int]:
        df = pd.read_excel('../crawled_data.xlsx', skiprows=0, usecols='C').values.tolist()
        return df, len(df)

    def get_term_index_from_list(self, term: str):
        termlist = np.array(self.index_file.iloc[:, 0])
        return np.where(termlist == term)[0][0]

    def get_term_count(self) -> int:
        return self.index_file.shape[0] - 2

    def get_indexed_documents_list_and_count(self) -> object:
        values = self.index_file.iloc[-1:-3:-1].values
        self.indexed_documents_list = ast.literal_eval(values[0][1])
        self.indexed_documents_count = values[1][1]

        return self.indexed_documents_list, self.indexed_documents_count
