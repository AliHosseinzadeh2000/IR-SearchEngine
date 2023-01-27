import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re


class Normalizer:

    def __init__(self):
        self.porter_stemmer = PorterStemmer()
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.stop_words = self.get_stop_words_from_file('../resources/stop_words_english.txt')
        nltk.download('wordnet')

    @staticmethod
    def get_stop_words_from_file(filename: str) -> list[str]:
        with open(filename, 'r', encoding='utf-8') as file:
            stop_words = file.read().splitlines()

            return stop_words

    def normalize_word_list(self, word_list) -> list[str]:
        normal_word_list = []
        for word in word_list:
            if word in self.stop_words:
                continue
            normal_word_list.append(self.normalize_a_word(word.lower()))
        return normal_word_list

    def normalize_a_word(self, word: str) -> str:
        word = self.remove_digits_from_word(word)
        word = self.stem_a_word(word)
        word = self.lemmatize_a_word(word)

        return word

    def stem_a_word(self, word: str) -> str:
        try:
            word = self.porter_stemmer.stem(word)
        except Exception:
            pass
        return word

    def lemmatize_a_word(self, word: str) -> str:
        try:
            word = self.wordnet_lemmatizer.lemmatize(word)
        except Exception:
            pass
        return word

    def remove_digits_from_word(self, word: str) -> str:
        return re.sub(r'\d+', '', word)
