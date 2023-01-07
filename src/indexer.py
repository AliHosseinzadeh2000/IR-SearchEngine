import PyPDF2
import re
import time


def main():
    stop_words = get_stop_words_from_file('../resources/stop_words_english.txt')
    doc_pages = parse_document('../repo/_Ian_Sommerville_Engineering_software_products_An_Introduction_to.pdf')

    start_time = time.time()
    final_index = {}

    for page in doc_pages:
        page_content = page.extract_text()
        final_index = index_a_page(final_index, page_content, stop_words)

    print(final_index)
    print(len(final_index))
    print(f'######## it took {time.time() - start_time}s to finish ########')


def get_stop_words_from_file(filename: str) -> list:
    with open(filename, 'r', encoding='utf-8') as file:
        stop_words = file.read().splitlines()

        return stop_words


def parse_document(path_to_doc: str):
    reader = PyPDF2.PdfReader(path_to_doc)
    return reader.pages


def index_a_page(final_index: dict, page_content: str, stop_words: list):
    for word in get_words_from_text(page_content):
        if word.lower() in stop_words:
            continue

        if not word.lower() in final_index.keys():
            final_index[word.lower()] = [1]
        else:
            final_index[word.lower()][0] += 1

    return final_index


def get_words_from_text(text: str):
    return re.findall(r'\w+', text)


main()
