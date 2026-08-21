import traceback

print("--- STARTING TRAINING SCRIPT ---")

try:
    print("1. Importing libraries...")
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from preprocess import preprocess_text
    
    print("2. Loading news.csv...")
    df = pd.read_csv("news.csv")
    df = df.dropna(subset=['text'])
    
    print("3. Preprocessing text (This may take a minute...)")
    # Adding a safeguard to drop completely empty strings after cleaning
    df["text"] = df["text"].astype(str).apply(preprocess_text)
    df = df[df["text"].str.strip() != ""] 
    
    print("4. Splitting data & Vectorizing...")
    X = df["text"]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    
    print("5. Training the Naive Bayes Model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    print("6. Saving .joblib files to root directory...")
    joblib.dump(model, "fake_news_model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    
    print("✅ SUCCESS! Models generated.")

except Exception as e:
    print("\n❌ CRASH DETECTED!")
    print("Here is the exact error:")
    print("-" * 40)
    traceback.print_exc()
    print("-" * 40)