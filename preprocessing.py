import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List

def _download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Initialize resources once at module level
_download_nltk_resources()

class TextPreprocessor:
    """
    Handles text cleaning and normalization for the Information Retrieval module.
    Provides methods to transform raw news articles and queries into cleaned tokens.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Perform a full preprocessing pipeline on the input text.

        Pipeline:
        1. Lowercase
        2. Remove HTML tags
        3. Remove URLs
        4. Remove punctuation
        5. Remove numbers
        6. Tokenization
        7. Stopword removal
        8. Lemmatization

        Args:
            text (str): The raw input string.

        Returns:
            str: The cleaned and normalized string.
        """
        if not text:
            return ""

        # 1. Lowercase
        text = text.lower()

        # 2. Remove HTML tags
        text = re.sub(r'<[^>]*>', ' ', text)

        # 3. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)

        # 4. Remove punctuation
        # We replace punctuation with space to avoid merging words
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

        # 5. Remove numbers
        text = re.sub(r'\d+', ' ', text)

        # 6. Tokenize
        tokens = word_tokenize(text)

        # 7. Remove stopwords and 8. Lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and word.strip() != ""
        ]

        return " ".join(cleaned_tokens)

    def preprocess_documents(self, documents: List[str]) -> List[str]:
        """
        Preprocess a list of documents.

        Args:
            documents (List[str]): A list of raw document texts.

        Returns:
            List[str]: A list of cleaned document texts.
        """
        return [self.clean_text(doc) for doc in documents]

if __name__ == "__main__":
    # Quick test for preprocessing logic
    sample_text = "The <b>Quick</b> brown fox jumps over the lazy dog! Visit https://example.com for more 123 info."
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned:  {cleaned}")
