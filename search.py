import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from preprocessing import TextPreprocessor
from vectorizer import IRVectorizer
from indexer import InvertedIndex

class SearchEngine:
    """
    Orchestrates the Information Retrieval pipeline.
    Combines preprocessing, vectorization, and the inverted index to rank documents.
    """

    def __init__(self, vectorizer_path: str, matrix_path: str, index_path: str):
        """
        Initializes the search engine by loading the trained models and index.

        Args:
            vectorizer_path (str): Path to the saved TfidfVectorizer.
            matrix_path (str): Path to the saved TF-IDF matrix.
            index_path (str): Path to the saved Inverted Index.
        """
        import os
        for path in [vectorizer_path, matrix_path, index_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found at {path}. Please run build_index.py first.")

        self.preprocessor = TextPreprocessor()

        self.vectorizer = IRVectorizer()
        self.vectorizer.load_vectorizer(vectorizer_path, matrix_path)

        self.indexer = InvertedIndex()
        self.indexer.load(index_path)

    def search(self, query: str, top_k: int = 5, doc_metadata: Dict[int, Dict] = None) -> List[Dict[str, Any]]:
        """
        Pipeline:
        query -> preprocess -> vectorizer.transform -> cosine_similarity -> rank -> Top-K

        Args:
            query (str): The raw user query string.
            top_k (int): Number of top results to return.
            doc_metadata (Dict[int, Dict]): Mapping of doc index to metadata (e.g., title, content).

        Returns:
            List[Dict]: A list of the top-K documents with their scores.
        """
        # 1. Preprocess the query
        cleaned_query = self.preprocessor.clean_text(query)

        # 2. Transform query into a TF-IDF vector
        query_vector = self.vectorizer.transform_query(cleaned_query)

        # 3. Compute Cosine Similarity against the document matrix
        # self.vectorizer.tfidf_matrix is the matrix of all documents
        similarities = cosine_similarity(query_vector, self.vectorizer.tfidf_matrix).flatten()

        # 4. Sort descending and get Top-K indices
        # argsort returns indices that would sort the array; we take the last top_k and reverse them
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score <= 0:
                continue

            # Try to map index to document ID and metadata
            # The index in the TF-IDF matrix corresponds to the order of docs provided during build
            doc_id = str(idx) # Default fallback
            title = "Unknown Title"

            if doc_metadata and idx in doc_metadata:
                meta = doc_metadata[idx]
                doc_id = meta.get('id', doc_id)
                title = meta.get('title', title)

            results.append({
                "id": doc_id,
                "title": title,
                "score": round(score, 4)
            })

        return results

if __name__ == "__main__":
    # This block is for structural verification.
    # In real usage, it requires saved .joblib files.
    print("SearchEngine module loaded. Integration tests are handled in test_search.py.")
