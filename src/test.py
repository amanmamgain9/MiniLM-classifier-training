import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# Load model
model = AutoModelForSequenceClassification.from_pretrained('./finetuned_minilm')
tokenizer = AutoTokenizer.from_pretrained('./finetuned_minilm')

# Load label map
with open('./finetuned_minilm/label_map.json', 'r') as f:
    label_map = json.load(f)
id2label = {v: k for k, v in label_map.items()}

# Test
def classify(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    outputs = model(**inputs)
    pred = outputs.logits.argmax(-1).item()
    return id2label[pred]

# Examples
texts = [
    "Buy groceries tomorrow",
    "Weekly team meeting every Monday",
    "Drink 8 glasses of water daily",
    "Launch new product by Q2"
]

for text in texts:
    print(f"{text} -> {classify(text)}")