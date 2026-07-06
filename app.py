import streamlit as st
import time # We import time to fake a delay for the loading animation
from mock_backend import get_search_results

st.set_page_config(page_title="Smart News", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("Settings")
data_source = st.sidebar.radio("Select Data Source:", ["Offline (30-day Dataset)", "Live (NewsAPI)"])
selected_category = st.sidebar.selectbox("Filter by Category:", ["All", "Technology", "Finance", "Politics"])

st.sidebar.divider()

# --- TEAM CREDITS ---
st.sidebar.caption("👨‍💻 **Developed by:**")
st.sidebar.caption("• Frontend: Dhruv")
st.sidebar.caption("• Search Engine: Aayush")
st.sidebar.caption("• Data & API: Lohitaksha")
st.sidebar.caption("• Credibility ML: Akshay")
st.sidebar.caption("• Summarizer NLP: Shiva")

# --- HEADER ---
st.title("📰 Smart News Summarizer & Credibility Analyzer")
st.write("Search for news and get AI-summarized, credibility-checked results.")

# --- MAIN SEARCH BAR ---
search_query = st.text_input("Enter a news topic (e.g., 'AI', 'Markets'):")

if st.button("Search"):
    if search_query:
        st.write(f"Searching for: **{search_query}** using {data_source}...")
        st.divider() 
        
        # --- LOADING ANIMATION ---
        with st.spinner('Scraping news, analyzing credibility, and generating AI summaries...'):
            # We add a fake 2-second delay to test the spinner
            time.sleep(2) 
            
            # This is where your teammates' real functions will go
            results = get_search_results(search_query)
        
        # --- DISPLAY RESULTS ---
        for article in results:
            if selected_category == "All" or article["category"] == selected_category:
                
                col1, col2 = st.columns([3, 1]) 
                with col1:
                    st.subheader(article["title"])
                    st.caption(f"Source: {article['source']} | Published: {article['time']} | Category: {article['category']}")
                    with st.expander("Read AI Summary"):
                        st.write(article["summary"])
                
                with col2:
                    score = article["credibility"]
                    st.metric(label="Credibility", value=f"{score}%")
                    if score >= 80:
                        st.success("High Credibility")
                    elif score >= 60:
                        st.warning("Medium Credibility")
                    else:
                        st.error("Low Credibility")
                
                st.divider() 
    else:
        st.warning("Please enter a search term first.")