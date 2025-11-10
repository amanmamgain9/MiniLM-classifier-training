import json
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Load ONNX models
model_orig = ORTModelForSequenceClassification.from_pretrained("./onnx_model")
model_quant = ORTModelForSequenceClassification.from_pretrained("./onnx_quantized")
tokenizer = AutoTokenizer.from_pretrained("./onnx_model")

# Load label map
with open("./finetuned_minilm/label_map.json", "r") as f:
    label_map = json.load(f)
id2label = {v: k for k, v in label_map.items()}


# Test function
def classify(text, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    pred = outputs.logits.argmax(-1).item()
    return id2label[pred]


# Test cases
texts = [
    "Buy groceries tomorrow",
    "Weekly team meeting every Monday",
    "Drink 8 glasses of water daily",
    "Launch new product by Q2",
    "2025-12-29的医生预约",
    "每天早上跑步",
]

print("Original ONNX vs Quantized:")
for text in texts:
    orig = classify(text, model_orig)
    quant = classify(text, model_quant)
    match = "✓" if orig == quant else "✗"
    print(f"{match} {text[:40]:40} | Orig: {orig:15} | Quant: {quant}")
