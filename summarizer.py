import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# --- CLOUD-SAFE NLTK DOWNLOADS ---
# We check for and download each package independently to prevent skipping one if the other exists.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
# ---------------------------------

def summarize_text(text, sentences_count=3):
    try:
        # Safety check for very short text
        if not text or len(str(text).strip()) < 20:
            return text
            
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentences_count)

        result = " ".join(str(sentence) for sentence in summary)

        if not result:
            return text 

        return result

    except Exception as e:
        return f"Summarization Error: {e}"