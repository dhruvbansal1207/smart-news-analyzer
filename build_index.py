import os
import pandas as pd
from preprocessing import TextPreprocessor
from indexer import InvertedIndex
from vectorizer import IRVectorizer

def build_index(csv_path: str, model_dir: str = "models"):
    """
    Main workflow to build the IR system from a CSV dataset.

    Workflow:
    1. Load CSV (expects 'id', 'title', 'content' columns)
    2. Preprocess documents
    3. Build Inverted Index
    4. Build TF-IDF Model
    5. Save all models to disk
    """
    # Ensure the models directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    # 1. Load CSV
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if df.empty:
        print("CSV is empty. Nothing to index.")
        return

    # We assume the CSV has 'id', 'title', and 'content' columns
    # If 'content' is missing, we can use 'title' as the text source
    text_column = 'content' if 'content' in df.columns else 'title'

    # 2. Preprocess
    print("Preprocessing documents...")
    preprocessor = TextPreprocessor()
    # We combine title and content for better indexing if both exist
    combined_text = df.apply(lambda row: f"{row.get('title', '')} {row.get('content', '')}", axis=1)
    cleaned_docs = preprocessor.preprocess_documents(combined_text.tolist())

    # 3. Build Inverted Index
    print("Building Inverted Index...")
    indexer = InvertedIndex()
    # Create a list of {id, text} for the indexer
    docs_for_index = [
        {"id": str(df.iloc[i]['id']), "text": cleaned_docs[i]}
        for i in range(len(cleaned_docs))
    ]
    indexer.build_index(docs_for_index)

    # 4. Build TF-IDF Model
    print("Building TF-IDF Model...")
    vectorizer = IRVectorizer()
    vectorizer.build_vectorizer(cleaned_docs)

    # 5. Save Models
    index_path = os.path.join(model_dir, "index.joblib")
    vec_path = os.path.join(model_dir, "vectorizer.joblib")
    tfidf_path = os.path.join(model_dir, "tfidf.joblib")

    indexer.save(index_path)
    vectorizer.save_vectorizer(vec_path, tfidf_path)

    print(f"\nIndexing complete!")
    print(f"Models saved to {model_dir}:")
    print(f"- {index_path}")
    print(f"- {vec_path}")
    print(f"- {tfidf_path}")

if __name__ == "__main__":
    # To run this, you need a news_data.csv file with at least an 'id' column
    # Create a dummy CSV for demonstration if it doesn't exist
    csv_filename = "news_data.csv"
    if not os.path.exists(csv_filename):
        print(f"Creating dummy {csv_filename} for demonstration...")
        dummy_data = {
            "id": [1, 2, 3, 4, 5],
            "title": ["AI Breakthrough", "Climate Change", "Space Exploration", "Healthy Eating", "Global Economy"],
            "content": [
                "Artificial intelligence is advancing rapidly with new LLMs.",
                "The global temperature is rising due to greenhouse gases.",
                "NASA is planning a new mission to Mars for the next decade.",
                "Eating vegetables and fruits daily improves heart health.",
                "Inflation is impacting the global economy and trade."
            ]
        }
        pd.DataFrame(dummy_data).to_csv(csv_filename, index=False)

    build_index(csv_filename)
