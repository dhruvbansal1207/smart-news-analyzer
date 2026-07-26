import joblib
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class IRVectorizer:
    """
    Handles the TF-IDF vectorization process for the IR module.
    Responsible for building the TF-IDF model, persisting it, and transforming queries.
    """

    def __init__(self):
        # TfidfVectorizer with standard settings (lowercase is True by default)
        # We use stop_words=None because we handle stopwords in preprocessing.py
        self.vectorizer = TfidfVectorizer(stop_words=None)
        self.tfidf_matrix = None

    def build_vectorizer(self, documents: List[str]):
        """
        Fits the TF-IDF Vectorizer on the provided documents and computes the matrix.

        Args:
            documents (List[str]): A list of preprocessed document texts.
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        return self.tfidf_matrix

    def save_vectorizer(self, vectorizer_path: str, matrix_path: str):
        """
        Persists both the TF-IDF Vectorizer and the computed TF-IDF matrix to disk.

        Args:
            vectorizer_path (str): Path to save the vectorizer object.
            matrix_path (str): Path to save the TF-IDF matrix.
        """
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.tfidf_matrix, matrix_path)

    def load_vectorizer(self, vectorizer_path: str, matrix_path: str):
        """
        Loads the TF-IDF Vectorizer and TF-IDF matrix from disk.

        Args:
            vectorizer_path (str): Path to load the vectorizer object.
            matrix_path (str): Path to load the TF-IDF matrix.
        """
        self.vectorizer = joblib.load(vectorizer_path)
        self.tfidf_matrix = joblib.load(matrix_path)

    def transform_query(self, query: str) -> np.ndarray:
        """
        Transforms a preprocessed query string into a TF-IDF vector.

        Args:
            query (str): Preprocessed query text.

        Returns:
            np.ndarray: The resulting TF-IDF vector for the query.
        """
        # transform returns a sparse matrix; we convert to dense array for easier similarity calculation
        return self.vectorizer.transform([query]).toarray()

if __name__ == "__main__":
    # Quick test for vectorizer logic
    docs = [
        "the quick brown fox",
        "jumped over the lazy dog",
        "the quick brown dog"
    ]

    vec = IRVectorizer()
    matrix = vec.build_vectorizer(docs)
    print(f"Matrix shape: {matrix.shape}") # (3, 8) approx

    query_vec = vec.transform_query("quick brown dog")
    print(f"Query vector shape: {query_vec.shape}") # (1, 8)

    # Test persistence
    import os
    v_path, m_path = "test_vec.joblib", "test_mat.joblib"
    vec.save_vectorizer(v_path, m_path)

    new_vec = IRVectorizer()
    new_vec.load_vectorizer(v_path, m_path)
    print(f"Loaded matrix shape: {new_vec.tfidf_matrix.shape}")

    # Cleanup
    os.remove(v_path)
    os.remove(m_path)
