import streamlit as st
import time
from data_api import get_live_news 
from search_engine import get_offline_search_results # NEW: Import the offline engine

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
                # Use the Scikit-Learn search engine on the CSV
                results = get_offline_search_results(search_query)
            else:
                # Use the NewsAPI internet pipeline
                api_key = st.secrets["NEWS_API_KEY"]
                results = get_live_news(search_query, api_key)
        
        # --- DISPLAY RESULTS ---
        if not results:
            st.error("No news found for this topic. Try another search term.")
        else:
            for article in results:
                if selected_category == "All" or article["category"] == selected_category:
                    
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
                        st.info("Pending ML Model")
                    
                    st.divider() 
    else:
        st.warning("Please enter a search term first.")