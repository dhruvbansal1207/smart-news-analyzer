from newsapi import NewsApiClient

API_KEY = "55e3bd344ba74e34a3b66a31f6f06237"
newsapi = NewsApiClient(api_key=API_KEY)

def fetch_news(query, page_size=10):
    """Fetches live news articles from NewsAPI."""
    response = newsapi.get_everything(
        q=query,
        language="en",
        sort_by="publishedAt",
        page_size=page_size
    )

    articles = []
    for article in response["articles"]:
        articles.append({
            "title": article["title"],
            "content": article["content"],
            "description": article["description"],
            "source": article["source"]["name"],
            "date": article["publishedAt"],
            "url": article["url"]
        })

    return articles