import re
import unicodedata

TEXT_COLUMN = "review"
TARGET_COLUMN = "sentiment"
POSITIVE_LABEL = "positive"
TOKEN_PATTERN = r"(?u)\b\w[\w']+\b"
MIN_WORDS = 3
MAX_CHARACTERS = 20000

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"http\S+|www\.\S+")
_NOT_TEXT = re.compile(r"[^a-z0-9'\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _NOT_TEXT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def tokenize(text):
    if not isinstance(text, str):
        return []
    return re.findall(TOKEN_PATTERN, clean_text(text))


def find_signal_terms(text, term_weights, limit=8):
    if not isinstance(term_weights, dict):
        return [], []
    tokens = tokenize(text)
    if not tokens:
        return [], []
    candidates = set(tokens) | {f"{first} {second}" for first, second in zip(tokens, tokens[1:])}
    found = [(term, term_weights[term]) for term in candidates if term in term_weights]
    positive = sorted((item for item in found if item[1] > 0), key=lambda item: -item[1])[:limit]
    negative = sorted((item for item in found if item[1] < 0), key=lambda item: item[1])[:limit]
    return positive, negative
