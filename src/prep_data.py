import pandas as pd
from sklearn.model_selection import train_test_split

# Label mapping
label_map = {
    "tasks": 0,
    "events": 1,
    "habits": 2,
    "recurring_events": 3,
    "goals": 4,
    "projects": 5,
}

data = []
for file in [
    "tasks_400_english_400_chinese.csv",
    "events_400_english_400_chinese.csv",
    "habits_400_english_400_chinese.csv",
    "recurring_events_400_english_400_chinese.csv",
    "goals_400_english_400_chinese.csv",
    "projects_400_english_400_chinese.csv",
]:

    df = pd.read_csv(f"data/{file}")
    category = file.replace("_400_english_400_chinese.csv", "")
    text_col = df.columns[0]

    for text in df[text_col]:
        data.append({"text": str(text), "label": label_map[category]})

df_all = pd.DataFrame(data)

# Shuffle and split
train_df, val_df = train_test_split(
    df_all, test_size=0.2, random_state=42, stratify=df_all["label"], shuffle=True
)

# Save to data folder
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)

print(f"Train: {len(train_df)}, Val: {len(val_df)}")
print(train_df["label"].value_counts())
