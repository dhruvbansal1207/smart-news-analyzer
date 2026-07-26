import re
from news_api import fetch_news
from search_engine import get_offline_search_results
from credibility_model import get_credibility_score

def get_news(query, mode="online"):
    """
    Master Router combining the entire team's work.
    Routes to either NewsAPI or the Inverted Index, applying ML and NLP.
    """
    if mode.lower() == "online":
        # 1. Fetch from Lohitaksha's API
        raw_articles = fetch_news(query)
        formatted_results = []
        
        for article in raw_articles:
            raw_text = str(article.get("content") or article.get("description") or "")
            article_title = str(article.get("title", "No Title"))
            
            # 2. Clean messy API text (HTML & Truncation tags)
            clean_text = re.sub(r'<[^>]*>', '', raw_text)
            clean_text = clean_text.split("[+")[0].strip() + "..."
            
            # 3. Format to Streamlit UI Contract and apply Akshay's ML Model
            formatted_results.append({
                "title": article_title,
                "source": article.get("source", "Unknown Source"),
                "time": str(article.get("date", "Unknown Date"))[:10],
                "full_text": clean_text,
                "summary": clean_text, # Live text is too short for Shiva's TextRank
                "credibility": get_credibility_score(article_title), 
                "category": "Live Global News" 
            })
        return formatted_results

    elif mode.lower() == "offline":
        # Route directly to Aayush's Advanced Inverted Index & Shiva's Summarizer
        return get_offline_search_results(query)

    else:
        raise ValueError("Mode must be 'online' or 'offline'")