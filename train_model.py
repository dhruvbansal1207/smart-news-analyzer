import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

print("Loading Fake and True news datasets...")
# Load the data from your data folder
try:
    fake_df = pd.read_csv("data/Fake.csv")
    true_df = pd.read_csv("data/True.csv")
except Exception as e:
    print(f"Error loading CSVs. Make sure Fake.csv and True.csv are in the 'data/' folder. Details: {e}")
    exit()

# Add labels: 0 for Fake, 1 for Real
fake_df['label'] = 0
true_df['label'] = 1

# Combine them into one massive dataset and shuffle it
print("Combining and shuffling data...")
df = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# We will train the model to detect fake news based on the 'title' column
# Drop any blank titles to prevent crashes
df = df.dropna(subset=['title'])
X = df['title'] 
y = df['label']

print("Vectorizing text (converting words to math)...")
# Limit to top 5000 words so it runs fast and keeps the app lightweight
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

print("Training the Machine Learning Model...")
model = LogisticRegression()
model.fit(X_vectorized, y)

print("Saving AI brain to disk...")
# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save the trained components so credibility_model.py can use them
joblib.dump(model, 'models/credibility_model.joblib')
joblib.dump(vectorizer, 'models/credibility_vectorizer.joblib')

print("✅ Credibility Model trained and saved successfully!")