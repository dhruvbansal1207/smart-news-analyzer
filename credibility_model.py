import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier

def train_credibility_model():
    # 1. SYNTHETIC TRAINING DATA
    # We train the model on examples of credible news (1) vs. clickbait/sensationalism (0)
    data = {
        "headline": [
            "Federal Reserve Increases Interest Rates by 0.5%",
            "You Won't Believe What Happened Next!!!",
            "New Study Shows Water is Good For You",
            "SHOCKING TRUTH About Your Diet!!!",
            "Apple Announces New iPhone 15 Specifications",
            "Top 10 SECRETS They Don't Want You To Know",
            "Global Markets Fall Amidst Inflation Fears",
            "Is The World Ending?! Watch This Video Now!",
            "Local Mayor Re-elected for Second Term",
            "15 Ways to Get Rich QUICK! #3 Will Shock You!"
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] 
    }
    df = pd.DataFrame(data)

    # 2. FEATURE ENGINEERING
    # Computers can't read clickbait, so we convert linguistic red flags into numbers
    def extract_features(text):
        return {
            "length": len(text),
            "exclamations": text.count("!"),
            "questions": text.count("?"),
            "uppercase_words": sum(1 for word in str(text).split() if word.isupper() and len(word) > 1),
            "clickbait_phrases": 1 if re.search(r'(shocking|unbelievable|secrets|you won\'t believe|quick)', str(text).lower()) else 0
        }

    # Extract math features for our training data
    features_df = pd.DataFrame([extract_features(text) for text in df['headline']])

    # 3. TRAIN THE RANDOM FOREST ALGORITHM
    model = RandomForestClassifier(random_state=42, n_estimators=50)
    model.fit(features_df, df['label'])
    
    return model

# Train the model once when the file loads
rf_model = train_credibility_model()

# 4. THE INFERENCE FUNCTION (This is what our app will call)
def get_credibility_score(headline):
    try:
        # Extract features from the newly searched headline
        def extract_features(text):
            return {
                "length": len(text),
                "exclamations": text.count("!"),
                "questions": text.count("?"),
                "uppercase_words": sum(1 for word in str(text).split() if word.isupper() and len(word) > 1),
                "clickbait_phrases": 1 if re.search(r'(shocking|unbelievable|secrets|you won\'t believe|quick)', str(text).lower()) else 0
            }
            
        features = pd.DataFrame([extract_features(headline)])
        
        # Predict probability (returns an array like [0.2, 0.8] -> 80% credible)
        probability = rf_model.predict_proba(features)[0][1]
        
        # Convert to a 0-100 score format
        return int(probability * 100)
    except Exception as e:
        return 50 # Safe fallback if the model encounters weird text