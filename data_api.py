import requests
from summarizer import get_ai_summary # NEW: Import your NLP brain

def get_live_news(query, api_key):
    """Fetches live news from NewsAPI and formats it for the Streamlit frontend."""
    
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&language=en&sortBy=relevancy&pageSize=5"
    
    response = requests.get(url)
    data = response.json()
    
    formatted_articles = []
    
    if data.get("status") == "ok":
        for article in data["articles"]:
            
            raw_text = article.get("content") or article.get("description") or "No text available."
            
            # NEW: Run the raw text through your summarizer
            ai_summary = get_ai_summary(raw_text)
            
            formatted_articles.append({
                "title": article.get("title", "No Title"),
                "source": article["source"].get("name", "Unknown Source"),
                "time": article.get("publishedAt", "")[:10], 
                "full_text": raw_text,
                "summary": ai_summary, # Plug the output into the dictionary!
                "credibility": 50, 
                "category": "All" 
            })
            
    return formatted_articles