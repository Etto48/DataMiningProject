import os
import regex as re

PROJECT_ROOT = "C:/Users/the_E/Documents/APPUNTI/AIDE 1/Data mining and machine learning/Project/"
if not os.path.exists(PROJECT_ROOT):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
DOCS_PATH = os.path.join(PROJECT_ROOT, "docs")
DOCS_PAPER = os.path.join(DOCS_PATH, "paper")
PAPER_IMAGES = os.path.join(DOCS_PAPER, "images")
    
CLASSES = ['worry', 'sadness', 'love', 'happiness', 'surprise', 'anger', 'neutral']

_url_regex = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)'
_twitter_at_regex = r'@[A-Za-z0-9_]+'
_symbols = r'[^a-zA-ZÀ-ÿ\s]'
EXCLUDE_REGEX = re.compile(rf"({'|'.join([_url_regex, _twitter_at_regex, _symbols])})")