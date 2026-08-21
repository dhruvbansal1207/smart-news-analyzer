import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess_text

class SimilarityChecker:
    def __init__(self, trusted_news_path="trusted_news.csv"):
        self.df = pd.read_csv(trusted_news_path)
        if "text" not in self.df.columns:
            raise ValueError("trusted_news.csv must contain a 'text' column.")
            
        # Sample top 2000 rows to ensure fast startup and low RAM usage
        if len(self.df) > 2000:
            self.df = self.df.head(2000)

        self.df["text"] = self.df["text"].fillna("").astype(str).apply(preprocess_text)
        self.df = self.df[self.df["text"].str.strip().astype(bool)].reset_index(drop=True)
        
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
        self.trusted_vectors = self.vectorizer.fit_transform(self.df["text"])

    def calculate_similarity(self, article: str) -> dict:
        if not article or not isinstance(article, str):
            return {"similarity_score": 0.0, "matched_article": ""}
            
        cleaned = preprocess_text(article)
        if not cleaned.strip():
            return {"similarity_score": 0.0, "matched_article": ""}
            
        article_vector = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(article_vector, self.trusted_vectors)[0]
        best_index = similarities.argmax()
        
        return {
            "similarity_score": round(float(similarities[best_index]) * 100, 2),
            "matched_article": self.df.iloc[best_index]["text"]
        }