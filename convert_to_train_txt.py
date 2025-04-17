import pandas as pd

df = pd.read_csv("data/multipleChoiceResponses.csv", low_memory=False)
output_file = "data/train.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for col in df.columns[1:]:  # skip timestamp
        responses = df[col].dropna().unique()
        for resp in responses:
            if isinstance(resp, str) and len(resp.split()) > 3:
                f.write(f"{col.strip()}: {resp.strip()}\n")

print("✅ train.txt generated successfully.")
