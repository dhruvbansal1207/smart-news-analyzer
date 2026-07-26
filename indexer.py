import joblib
from collections import defaultdict
from typing import Dict, Set, List

class InvertedIndex:
    """
    Implements an inverted index for efficient document retrieval.
    Maps individual terms to the set of document IDs containing those terms.
    """

    def __init__(self):
        # Maps term (str) -> set of document IDs (str or int)
        self.index: Dict[str, Set] = defaultdict(set)

    def add_document(self, doc_id: str, text: str):
        """
        Indexes a single document.

        Args:
            doc_id (str): Unique identifier for the document.
            text (str): Preprocessed text of the document.
        """
        tokens = text.split()
        for token in tokens:
            self.index[token].add(doc_id)

    def build_index(self, documents: List[Dict]):
        """
        Builds the inverted index from a list of documents.

        Args:
            documents (List[Dict]): A list of dictionaries, each containing 'id' and 'text'.
        """
        for doc in documents:
            self.add_document(doc['id'], doc['text'])

    def search(self, term: str) -> Set:
        """
        Retrieves the set of document IDs containing the given term.

        Args:
            term (str): The term to search for.

        Returns:
            Set: A set of matching document IDs.
        """
        return self.index.get(term, set())

    def search_multiple(self, terms: List[str], operator: str = "AND") -> Set:
        """
        Retrieves document IDs based on multiple terms using Boolean logic.

        Args:
            terms (List[str]): List of terms to search.
            operator (str): "AND" (intersection) or "OR" (union). Defaults to "AND".

        Returns:
            Set: The resulting set of document IDs.
        """
        if not terms:
            return set()

        results = [self.search(term) for term in terms]

        if operator.upper() == "AND":
            # Intersection of all sets
            return set.intersection(*results) if results else set()
        elif operator.upper() == "OR":
            # Union of all sets
            return set.union(*results) if results else set()
        else:
            raise ValueError("Operator must be either 'AND' or 'OR'")

    def save(self, path: str):
        """
        Persists the inverted index to disk using joblib.

        Args:
            path (str): File path where the index should be saved.
        """
        joblib.dump(dict(self.index), path)

    def load(self, path: str):
        """
        Loads the inverted index from a file.

        Args:
            path (str): File path from which the index should be loaded.
        """
        loaded_index = joblib.load(path)
        self.index = defaultdict(set, loaded_index)

if __name__ == "__main__":
    # Quick test for indexer logic
    docs = [
        {"id": "doc1", "text": "the quick brown fox"},
        {"id": "doc2", "text": "jumped over the lazy dog"},
        {"id": "doc3", "text": "the quick brown dog"}
    ]

    idx = InvertedIndex()
    idx.build_index(docs)

    print(f"Search 'quick': {idx.search('quick')}")        # Expected: {'doc1', 'doc3'}
    print(f"Search 'dog': {idx.search('dog')}")            # Expected: {'doc2', 'doc3'}
    print(f"AND search ['quick', 'dog']: {idx.search_multiple(['quick', 'dog'], 'AND')}") # Expected: {'doc3'}
    print(f"OR search ['quick', 'dog']: {idx.search_multiple(['quick', 'dog'], 'OR')}")   # Expected: {'doc1', 'doc2', 'doc3'}
