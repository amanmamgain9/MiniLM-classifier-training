import pandas as pd
import re
import os

def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

os.makedirs("data/english", exist_ok=True)

files = [
    "tasks_400_english_400_chinese.csv",
    "events_400_english_400_chinese.csv",
    "habits_400_english_400_chinese.csv",
    "recurring_events_400_english_400_chinese.csv",
    "goals_400_english_400_chinese.csv",
    "projects_400_english_400_chinese.csv",
]

for file in files:
    df = pd.read_csv(f"data/{file}")
    text_col = df.columns[0]
    
    english_rows = [text for text in df[text_col] if not is_chinese(str(text))]
    
    df_english = pd.DataFrame({text_col: english_rows})
    
    output_name = file.replace("_400_english_400_chinese.csv", "_400_english.csv")
    df_english.to_csv(f"data/english/{output_name}", index=False)
    
    print(f"{output_name}: {len(df_english)} rows")