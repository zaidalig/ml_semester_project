import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load CSV file
data_path = "data/ml_qa_dataset_extended_15k.csv"
df = pd.read_csv(data_path)

# Combine question and answer into prompt-like text
df["text"] = df["question"].astype(str) + "\n\n" + df["answer"].astype(str)

# Split into train and valid
train_texts, valid_texts = train_test_split(df["text"], test_size=0.2, random_state=42)

# Write to files
os.makedirs("data", exist_ok=True)

with open("data/train.txt", "w", encoding="utf-8") as f:
    for line in train_texts:
        f.write(line.strip() + "\n\n")

with open("data/valid.txt", "w", encoding="utf-8") as f:
    for line in valid_texts:
        f.write(line.strip() + "\n\n")

print("✅ Converted CSV to train.txt and valid.txt")
