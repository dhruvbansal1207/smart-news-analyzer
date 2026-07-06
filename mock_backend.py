def get_search_results(query):
    # Added a 'category' key to each dictionary
    dummy_articles = [
        {
            "title": f"The Future of {query.title()}: What You Need to Know",
            "source": "Tech Insider",
            "time": "2 hours ago",
            "summary": f"This is an AI-generated summary about {query}. It extracts the key points from the original article.",
            "credibility": 92,
            "category": "Technology"
        },
        {
            "title": f"Global Markets React to {query.title()} Trends",
            "source": "Financial Times Fake",
            "time": "5 hours ago",
            "summary": f"Experts are weighing in on the economic impact of {query}.",
            "credibility": 68,
            "category": "Finance"
        },
        {
            "title": f"Debate Heats Up Over {query.title()} Legislation",
            "source": "Capital News",
            "time": "1 day ago",
            "summary": f"Lawmakers are clashing over the new policies regarding {query}.",
            "credibility": 45,
            "category": "Politics"
        }
    ]
    return dummy_articles