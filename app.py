import streamlit as st
import time
from data_manager import get_news # The single unified bridge

st.set_page_config(page_title="Smart News", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("Settings")
data_source = st.sidebar.radio("Select Data Source:", ["Offline (30-day Dataset)", "Live (NewsAPI)"])
selected_category = st.sidebar.selectbox("Filter by Category:", ["All", "Technology", "Finance", "Politics"])

st.sidebar.divider()

# --- TEAM CREDITS ---
st.sidebar.caption("👨‍💻 **Developed by:**")
st.sidebar.caption("• Frontend & API: Dhruv")
st.sidebar.caption("• Search Engine: Aayush")
st.sidebar.caption("• Data pipeline: Lohitaksha")
st.sidebar.caption("• Credibility ML: Akshay")
st.sidebar.caption("• Summarizer NLP: Shiva")

# --- HEADER ---
st.title("📰 Smart News Summarizer & Credibility Analyzer")
st.write("Search for live global news using your custom API pipeline.")

# --- MAIN SEARCH BAR ---
search_query = st.text_input("Enter a news topic (e.g., 'SpaceX', 'Apple'):")

if st.button("Search"):
    if search_query:
        st.write(f"Searching the web for: **{search_query}**...")
        st.divider() 
        
        # --- LOADING ANIMATION & DATA FETCHING ---
        with st.spinner('Fetching and analyzing articles...'):
            if data_source == "Offline (30-day Dataset)":
                # Route through the unified manager (Offline Mode)
                results = get_news(search_query, mode="offline")
            else:
                # Route through the unified manager (Online Mode)
                results = get_news(search_query, mode="online")
        
        # --- DISPLAY RESULTS ---
        if not results:
            st.error("No news found for this topic. Try another search term.")
        else:
            for article in results:
                # Use .get() safely in case category is missing from live news
                if selected_category == "All" or article.get("category", "All") == selected_category:
                    
                    col1, col2 = st.columns([3, 1]) 
                    with col1:
                        st.subheader(article["title"])
                        st.caption(f"Source: {article['source']} | Published: {article['time']}")
                        with st.expander("Read Text & Summary Status"):
                            st.write("**Raw Text Fetched:**")
                            st.write(article["full_text"])
                            st.divider()
                            st.write(f"**Summary Engine:** {article['summary']}")
                    
                    with col2:
                        score = article["credibility"]
                        st.metric(label="Credibility Score", value=f"{score}%")
                        # The "Pending ML Model" text has been removed!
                    
                    st.divider() 
    else:
        st.warning("Please enter a search term first.")