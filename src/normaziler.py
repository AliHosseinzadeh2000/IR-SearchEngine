import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re


class Normalizer:

    def __init__(self):
        self.porter_stemmer = PorterStemmer()
        self.wordnet_lemmatizer = WordNetLemmatizer()
        nltk.download('wordnet')

    def normalize_word_list(self, word_list) -> list[str]:
        return [self.normalize_a_word(word.lower()) for word in word_list]

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
