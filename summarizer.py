import re # We import regex to hunt down and destroy HTML tags
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(raw_text):
    """Sanitizes the text by stripping HTML and truncation markers."""
    # 1. Strip out any HTML tags like <ul> or <li>
    text = re.sub(r'<[^>]+>', ' ', raw_text)
    # 2. Strip out the NewsAPI truncation marker e.g., [+2622 chars]
    text = re.sub(r'\[\+\d+\s*chars\]', '', text)
    # 3. Clean up any weird double spaces
    return " ".join(text.split())

def get_ai_summary(text, target_sentences=3):
    """Reads text and extracts the most important sentences dynamically."""
    
    if not text or len(text.strip()) < 20:
        return "No meaningful text provided to summarize."
        
    cleaned_text = clean_text(text)
    
    try:
        parser = PlaintextParser.from_string(cleaned_text, Tokenizer("english"))
        
        # Count how many sentences the NLP parser actually found
        actual_sentences = len(parser.document.sentences)
        
        # If the text is super short (like NewsAPI's 200-char limit), just return the cleaned text
        if actual_sentences <= 2:
            return cleaned_text
            
        # Ensure we never ask the math model for more sentences than exist
        safe_count = min(target_sentences, actual_sentences - 1)
        
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, safe_count)
        
        final_summary = " ".join([str(sentence) for sentence in summary_sentences])
        return final_summary
        
    except Exception as e:
        # If it still fails, gracefully return the cleaned text instead of an error
        return cleaned_text