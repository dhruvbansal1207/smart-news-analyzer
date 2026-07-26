import pandas as pd
import streamlit as st
from search import SearchEngine
from summarizer import summarize_text
from credibility_model import get_credibility_score

# --- 1. CACHE THE TEAMMATE'S SEARCH ENGINE ---
@st.cache_resource
def init_engine():
    try:
        # Load the pre-computed models
        engine = SearchEngine("models/vectorizer.joblib", "models/tfidf.joblib", "models/index.joblib")
        
        # Load the CSV to map the engine's ID hits back to the full text
        df = pd.read_csv("news_data.csv").fillna("Unknown")
        
        # Build the metadata dictionary the teammate's search method expects
        doc_metadata = {}
        for i, row in df.iterrows():
            doc_metadata[i] = {
                "id": str(row["id"]),
                "title": str(row["title"]),
                "content": str(row["content"]),
                "source": str(row.get("source", "Unknown")), 
                "category": str(row.get("category", "General")),
                "time": str(row.get("time", "Unknown Date"))
            }
        return engine, doc_metadata
    except Exception as e:
        st.error(f"Engine Load Error: Please run build_index.py first! ({e})")
        return None, None

# --- 2. STREAMLIT UI CONTRACT ---
def get_offline_search_results(query):
    engine, metadata = init_engine()
    if not engine: 
        return []

    # Call your teammate's search method
    top_hits = engine.search(query, top_k=5, doc_metadata=metadata)
    
    formatted_results = []
    for hit in top_hits:
        doc_id = str(hit["id"])
        
        # Match the document ID back to our metadata
        meta = next((m for m in metadata.values() if str(m["id"]) == doc_id), None)
        if not meta: continue
        
        raw_text = meta["content"]
        
        formatted_results.append({
            "title": meta["title"],
            "source": meta["source"],
            "time": str(meta["time"])[:10],
            "full_text": raw_text,
            "summary": summarize_text(raw_text), 
            "credibility": get_credibility_score(meta['title']), 
            "category": meta["category"]
        })
        
    return formatted_results