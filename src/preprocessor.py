from __future__ import annotations
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import pickle
from tqdm import tqdm

class Preprocessor:
    def __init__(self, kind: str = 'tfidf'):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True) 
        nltk.download('averaged_perceptron_tagger', quiet=True)
        stopwords = set(nltk.corpus.stopwords.words('english'))
        match kind:
            case 'tfidf':
                self.vectorizer = TfidfVectorizer(stop_words=list(stopwords), lowercase=True, strip_accents='ascii')
            case 'count':
                self.vectorizer = CountVectorizer(stop_words=list(stopwords), lowercase=True, strip_accents='ascii')
            case 'binary':
                self.vectorizer = CountVectorizer(stop_words=list(stopwords), lowercase=True, strip_accents='ascii', binary=True)
            case _:
                raise ValueError("Invalid kind of vectorizer. Choose between 'tfidf', 'count' and 'binary'.")
        
        self.kind = kind
        self.lemmatizer = WordNetLemmatizer()
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
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        stemmed_words = [self.stemmer.stem(word) for word in lemmatized_words]
        ret = " ".join(stemmed_words)
        
        return ret
        
    def fit(self, x: list[str]):
        clean_x = [self._preprocess_text(text) for text in tqdm(x, "Preprocessing data")]
        self.vectorizer.fit(clean_x)
        self.ready = True
        
    def __call__(self, x: list[str]):
        if self.ready:
            clean_x = [self._preprocess_text(text) for text in x]
            return self.vectorizer.transform(clean_x)
        else:
            raise SyntaxError("Preprocessor must be fitted before calling it. Use the Preprocessor.fit(x) method.")
        
    def get_indices(self, x: list[str], pad_to: int = None) -> list[list[int]]:
        if self.ready:
            clean_x = [self._preprocess_text(text) for text in x]
            unk_index = len(self.vectorizer.vocabulary_)
            pad_index = unk_index + 1
            x_indices = [[self.vectorizer.vocabulary_.get(word, unk_index) for word in document.split()] for document in clean_x]
            if pad_to is not None:
                ret = []
                for document in x_indices:
                    if len(document) < pad_to:
                        document = document + [pad_index] * (pad_to - len(document))
                    else:
                        document = document[:pad_to]
                    assert len(document) == pad_to
                    ret.append(document)
                x_indices = ret
            return x_indices
        else:
            raise SyntaxError("Preprocessor must be fitted before calling it. Use the Preprocessor.fit(x) method.")
        
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.ready:
            with open(path, 'wb') as f:
                pickle.dump((self.vectorizer, self.kind), f)
        else:
            raise SyntaxError("Preprocessor must be fitted before saving it.")
                
    def load(path: str) -> Preprocessor:
        self = Preprocessor()
        with open(path, 'rb') as f:
            self.vectorizer, self.kind = pickle.load(f)
            self.ready = True
        return self
    
    def __len__(self):
        return len(self.vectorizer.vocabulary_)
            