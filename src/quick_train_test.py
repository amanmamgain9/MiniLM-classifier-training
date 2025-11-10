import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

# Label mapping
label_map = {
    "tasks": 0,
    "events": 1,
    "habits": 2,
    "recurring_events": 3,
    "goals": 4,
    "projects": 5,
}

# Load data from english folder
data = []
for file in [
    "tasks_400_english.csv",
    "events_400_english.csv",
    "habits_400_english.csv",
    "recurring_events_400_english.csv",
    "goals_400_english.csv",
    "projects_400_english.csv",
]:
    df = pd.read_csv(f"data/english/{file}")
    category = file.replace("_400_english.csv", "")
    text_col = df.columns[0]
    
    # Use ALL samples (not just 10)
    for text in df[text_col]:
        data.append({"text": str(text), "label": label_map[category]})

df_all = pd.DataFrame(data)
print(f"Total samples: {len(df_all)}")
print(df_all["label"].value_counts())

# Split
train_df, val_df = train_test_split(
    df_all, test_size=0.2, random_state=42, stratify=df_all["label"]
)

# Convert to HF datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load model
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# Better training settings
training_args = TrainingArguments(
    output_dir="./models/quick_test",
    num_train_epochs=5,  # ← More epochs
    per_device_train_batch_size=16,  # ← Smaller batch = more updates
    per_device_eval_batch_size=16,
    learning_rate=2e-5,  # ← Slightly lower LR
    eval_strategy="epoch",  # ← Evaluate each epoch
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
print("\n🚀 Training...")
trainer.train()

# Evaluate
print("\n📊 Final Evaluation...")
results = trainer.evaluate()
print(f"Accuracy: {results['eval_accuracy']:.3f}")
print(f"F1 Score: {results['eval_f1']:.3f}")

# Save
trainer.save_model("./models/quick_test")
print("\n✅ Model saved!")

# REPL Loop
id2label = {0: "task", 1: "event", 2: "habit", 3: "recurring_event", 4: "goal", 5: "project"}
model = model.to("cpu")
model.eval()

print("\n" + "="*50)
print("🎮 INTERACTIVE TEST MODE")
print("="*50)
print("Type your text to classify (or 'quit' to exit)\n")

while True:
    user_input = input("📝 Enter text: ").strip()
    
    if user_input.lower() in ["quit", "exit", "q"]:
        print("👋 Bye!")
        break
    
    if not user_input:
        continue
    
    # Predict
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        probs = torch.softmax(logits, dim=-1)[0]
    
    # Show result
    print(f"   → Prediction: {id2label[prediction]} ({probs[prediction]:.2%} confidence)")
    
    # Show top 3
    top3 = torch.topk(probs, 3)
    print(f"   Top 3: ", end="")
    print(" | ".join([f"{id2label[idx.item()]}: {prob:.1%}" for prob, idx in zip(top3.values, top3.indices)]))
    print()