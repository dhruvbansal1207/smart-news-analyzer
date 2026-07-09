import pandas as pd

print("Loading massive dataset...")
df = pd.read_json("News_Category_Dataset_v3.json", lines=True)

print("Slicing the first 10,000 articles...")
mini_df = df.head(10000)

print("Saving lightweight version...")
mini_df.to_json("mini_dataset.json", orient="records", lines=True)

print("Done! You can now delete shrink_data.py")