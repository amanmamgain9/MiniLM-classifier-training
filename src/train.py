import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# Load model and tokenizer
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

# Load datasets
dataset = load_dataset(
    "csv", data_files={"train": "data/train.csv", "validation": "data/val.csv"}
)


# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)


tokenized_dataset = dataset.map(tokenize, batched=True)


# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


# Training arguments
training_args = TrainingArguments(
    output_dir="./models/training_output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print(f"\nValidation Results: {results}")

# Set label mappings
id2label = {0: "task", 1: "event", 2: "habit", 3: "recurring_event", 4: "goal", 5: "project"}
label2id = {v: k for k, v in id2label.items()}
model.config.id2label = id2label
model.config.label2id = label2id

# Save
trainer.save_model("./models/finetuned_minilm")
tokenizer.save_pretrained("./models/finetuned_minilm")

import json

label_map = {"task": 0, "event": 1, "habit": 2, "recurring_event": 3, "goal": 4, "project": 5}
with open("./models/finetuned_minilm/label_map.json", "w") as f:
    json.dump(label_map, f)

print("\nModel saved to ./models/finetuned_minilm")