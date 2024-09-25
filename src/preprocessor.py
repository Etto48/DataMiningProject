from __future__ import annotations
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from dmml_project.stopwords import STOPWORDS
from nltk.stem import SnowballStemmer
import pickle
from tqdm import tqdm

from dmml_project import EXCLUDE_REGEX

class Preprocessor:
    def __init__(self, kind: str = 'tfidf'):
        #nltk.download('stopwords', quiet=True)
        #stopwords = set(nltk.corpus.stopwords.words('english'))
        stopwords = STOPWORDS
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
        self.stemmer = SnowballStemmer('english')
        self.ready = False
        self.exclude_regex = EXCLUDE_REGEX
        
    def _preprocess_text(self, text: str) -> str:
        ret = "".join(filter(lambda x: x.isprintable(), text))
        ret = self.exclude_regex.sub(" ", ret)
        
        words = ret.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        ret = " ".join(stemmed_words)
        
        return ret
        
    def fit(self, x: list[str], verbose: bool = True):
        clean_x = [self._preprocess_text(text) for text in tqdm(x, "Preprocessing data", disable=not verbose)]
        self.vectorizer.fit(clean_x)
        self.ready = True
        
    def __call__(self, x: list[str]):
        if self.ready:
            clean_x = [self._preprocess_text(text) for text in x]
            return self.vectorizer.transform(clean_x)
        else:
            raise SyntaxError("Preprocessor must be fitted before calling it. Use the Preprocessor.fit(x) method.")
        
    def get_indices(self, x: list[str]) -> list[list[int]]:
        if self.ready:
            clean_x = [self._preprocess_text(text) for text in x]
            unk_index = len(self.vectorizer.vocabulary_)
            pad_index = unk_index + 1
            x_indices = [[self.vectorizer.vocabulary_.get(word, unk_index) for word in document.split()] for document in clean_x]
            max_len = max(map(len, x_indices))
            ret = []
            for document in x_indices:
                if len(document) < max_len:
                    document = document + [pad_index] * (max_len - len(document))
                assert len(document) == max_len
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
            