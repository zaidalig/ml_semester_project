with open("data/train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

split = int(len(lines) * 0.8)

with open("data/train.txt", "w", encoding="utf-8") as f:
    f.writelines(lines[:split])

with open("data/valid.txt", "w", encoding="utf-8") as f:
    f.writelines(lines[split:])

print("✅ train.txt and valid.txt are ready.")
