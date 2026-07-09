import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from summarizer import get_ai_summary

# --- PERFORMANCE UPGRADE ---
# This decorator forces Streamlit to memorize this math so it doesn't freeze your app!
@st.cache_resource
def load_and_vectorize_data(file_path="mini_dataset.json"):
    try:
        # 1. Load the JSON dataset (lines=True is required for this specific dataset format)
        df = pd.read_json(file_path, lines=True)
        
        # 2. Prevent RAM crashes by taking the most recent 10,000 articles
        df = df.head(10000)
        
        # 3. Rename the JSON keys to perfectly match our Data Contract
        df = df.rename(columns={
            "headline": "title",
            "authors": "source",
            "date": "time",
            "short_description": "full_text"
        })
        
        # Clean up empty sources
        df['source'] = df['source'].replace('', 'Unknown Source')
        
        # 4. Do the heavy TF-IDF math ONCE
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['full_text'])
        
        return df, vectorizer, tfidf_matrix
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None, None

# --- THE SEARCH FUNCTION ---
def get_offline_search_results(query, file_path="mini_dataset.json"):
    
    # Grab the pre-calculated math from memory
    df, vectorizer, tfidf_matrix = load_and_vectorize_data(file_path)
    
    if df is None:
        return []
        
    # Convert the user's search query into a vector
    query_vector = vectorizer.transform([query])
    
    # Calculate Cosine Similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Add scores and filter out bad matches
    df['score'] = similarity_scores
    best_matches = df[df['score'] > 0.05].sort_values(by='score', ascending=False).head(5)
    
    # Package into our frontend dictionary contract
    formatted_results = []
    for index, row in best_matches.iterrows():
        raw_text = row['full_text']
        ai_summary = get_ai_summary(raw_text)
        
        formatted_results.append({
            "title": row['title'],
            "source": row['source'],
            "time": str(row['time'])[:10],
            "full_text": raw_text,
            "summary": ai_summary, 
            "credibility": 50, 
            "category": row['category']
        })
        
    return formatted_results