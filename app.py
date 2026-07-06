import streamlit as st
from mock_backend import get_search_results

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart News", layout="wide")

# --- HEADER ---
st.title("📰 Smart News Summarizer & Credibility Analyzer")
st.write("Search for news and get AI-summarized, credibility-checked results.")

# --- SIDEBAR ---
st.sidebar.header("Settings")
data_source = st.sidebar.radio("Select Data Source:", ["Offline (30-day Dataset)", "Live (NewsAPI)"])

# NEW: Add a dropdown menu to filter categories
selected_category = st.sidebar.selectbox("Filter by Category:", ["All", "Technology", "Finance", "Politics"])

# --- MAIN SEARCH BAR ---
search_query = st.text_input("Enter a news topic (e.g., 'AI', 'Markets'):")

if st.button("Search"):
    if search_query:
        st.write(f"Searching for: **{search_query}** using {data_source}...")
        st.divider() 

        results = get_search_results(search_query)
        
        # Loop through the results
        for article in results:
            
            # NEW: Only draw the UI if the article matches the chosen filter
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