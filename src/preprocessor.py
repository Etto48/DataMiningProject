from __future__ import annotations
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.stem import PorterStemmer
import re
import pickle

class Preprocessor:
    def __init__(self, kind: str = 'tfidf'):
        nltk.download('stopwords', quiet=True)
        stopwords = set(nltk.corpus.stopwords.words('english'))
        
        if kind == 'tfidf':
            self.vectorizer = TfidfVectorizer(stop_words=list(stopwords), lowercase=True, strip_accents='ascii')
        elif kind == 'count':
            self.vectorizer = CountVectorizer(stop_words=list(stopwords), lowercase=True, strip_accents='ascii')
        elif kind == 'binary':
            self.vectorizer = CountVectorizer(stop_words=list(stopwords), lowercase=True, strip_accents='ascii', binary=True)
        else:
            raise ValueError("Invalid kind of vectorizer. Choose between 'tfidf', 'count' and 'binary'.")
        
        self.stemmer = PorterStemmer()
        self.ready = False
        
    def _preprocess_text(self, text: str) -> str:
        # remove urls
        url_regex = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)'
        twitter_at_regex = r'@[A-Za-z0-9_]+'
        punctuation = r'[^\w\s]'
        symbols = r'[^a-zA-Z0-9\s]'
        ret = re.sub(url_regex, '', text)
        ret = re.sub(twitter_at_regex, '', ret)
        ret = re.sub(punctuation, " ", ret)
        ret = re.sub(symbols, " ", ret)
        # remove non ascii and non printable characters
        ret = "".join(filter(lambda x: x.isprintable() and x.isascii(), ret))
        
        words = ret.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        ret = " ".join(stemmed_words)
        
        return ret
        
    def fit(self, x: list[str]):
        clean_x = [self._preprocess_text(text) for text in x]
        self.vectorizer.fit(clean_x)
        self.ready = True
        
    def __call__(self, x: list[str]):
        if self.ready:
            clean_x = [self._preprocess_text(text) for text in x]
            return self.vectorizer.transform(clean_x)
        else:
            raise SyntaxError("Preprocessor must be fitted before calling it. Use the Preprocessor.fit(x) method.")
        
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.ready:
            with open(path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
        else:
            raise SyntaxError("Preprocessor must be fitted before saving it.")
                
    def load(path: str) -> Preprocessor:
        self = Preprocessor()
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
            self.ready = True
        return self
    
    def __len__(self):
        return len(self.vectorizer.vocabulary_)
            