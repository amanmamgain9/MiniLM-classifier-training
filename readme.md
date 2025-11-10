
It works but the data was one shot generated using chatgpt, second it is not clear what is the difference between project ,task and goals as these words are used interchangeably in normal conversation and language

# Text Classifier Training & Deployment
Multi-label text classifier for categorizing tasks, events, habits, recurring events, goals, and projects in English and Chinese.

## Setup
```bash
# Create project
mkdir classifier-training && cd classifier-training

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate optimum[onnxruntime-gpu] onnx scikit-learn pandas
```

## Data Preparation

Place CSV files in `data/` folder:
- tasks_400_english_400_chinese.csv
- events_400_english_400_chinese.csv
- habits_400_english_400_chinese.csv
- recurring_events_400_english_400_chinese.csv
- goals_400_english_400_chinese.csv
- projects_400_english_400_chinese.csv

Run prep script:
```bash
python src/prep_data.py
```

Creates `data/train.csv` and `data/val.csv` with 80/20 split.

## Training
```bash
python src/train.py
```

Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`  
Output: `./models/finetuned_minilm/`  
Accuracy: ~100% on validation set

**Note:** Training script automatically sets `id2label` and `label2id` in model config to ensure proper label mapping after ONNX export.

## ONNX Export & Quantization
```bash
# Export to ONNX
optimum-cli export onnx --model ./models/finetuned_minilm --task text-classification ./models/onnx_model

# Quantize (450MB → 113MB)
python -c "from optimum.onnxruntime import ORTQuantizer; from optimum.onnxruntime.configuration import AutoQuantizationConfig; quantizer = ORTQuantizer.from_pretrained('./models/onnx_model'); qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False); quantizer.quantize(save_dir='./models/onnx_quantized', quantization_config=qconfig)"

# Create required structure for browser loading
mkdir -p ./models/onnx_quantized/onnx
cp ./models/onnx_quantized/model_quantized.onnx ./models/onnx_quantized/onnx/
```

## Deploy to Cloudflare R2
```bash
# Install wrangler
npm install -g wrangler

# Login
wrangler login

# Create bucket
wrangler r2 bucket create text-classifier-model

# Upload files with onnx_quantized/ prefix
wrangler r2 object put text-classifier-model/onnx_quantized/config.json --file ./models/onnx_quantized/config.json --remote
wrangler r2 object put text-classifier-model/onnx_quantized/tokenizer.json --file ./models/onnx_quantized/tokenizer.json --remote
wrangler r2 object put text-classifier-model/onnx_quantized/tokenizer_config.json --file ./models/onnx_quantized/tokenizer_config.json --remote
wrangler r2 object put text-classifier-model/onnx_quantized/special_tokens_map.json --file ./models/onnx_quantized/special_tokens_map.json --remote
wrangler r2 object put text-classifier-model/onnx_quantized/onnx/model_quantized.onnx --file ./models/onnx_quantized/onnx/model_quantized.onnx --remote
```

## Cloudflare Worker CDN
```bash
cd r2-model-cdn
wrangler deploy
```

Worker URL: `https://r2-model-cdn.youraccount.workers.dev`

## Browser Integration
```html
<!DOCTYPE html>
<html>
<head>
    <title>Text Classifier</title>
</head>
<body>
    <h1>Text Classifier</h1>
    <input type="text" id="text" placeholder="Enter text..." style="width: 400px">
    <button onclick="classify()">Classify</button>
    <div id="result"></div>
    <div id="time"></div>

    <script type="module">
        import { env, pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

        // Configure custom CDN path
        env.localModelPath = 'https://r2-model-cdn.youraccount.workers.dev/';
        env.allowRemoteModels = false;

        let classifier;

        async function loadModel() {
            document.getElementById('result').textContent = 'Loading model...';
            classifier = await pipeline('text-classification', 'onnx_quantized');
            document.getElementById('result').textContent = 'Model loaded!';
        }

        window.classify = async function() {
            const text = document.getElementById('text').value;
            const start = performance.now();
            const result = await classifier(text);
            const time = performance.now() - start;
            
            document.getElementById('result').textContent = `Result: ${result[0].label}`;
            document.getElementById('time').textContent = `Time: ${time.toFixed(2)}ms`;
        }

        loadModel();
    </script>
</body>
</html>
```

## Testing

**Local:**
```bash
python -m http.server 8000
# Open http://localhost:8000
# Use: classifier = await pipeline('text-classification', './onnx_quantized/');
```

**CDN:**
```bash
# Verify files accessible
curl https://r2-model-cdn.youraccount.workers.dev/onnx_quantized/config.json

# Test in browser with env.localModelPath configuration
```

## Label Mapping
```json
{
  "task": 0,
  "event": 1,
  "habit": 2,
  "recurring_event": 3,
  "goal": 4,
  "project": 5
}
```

## Costs

- Training: Local GPU (free)
- R2 Storage: Free (under 10GB)
- R2 Bandwidth: Unlimited free egress
- Worker: 100k requests/day free

## Project Structure
```
classifier-training/
├── data/                    # Training data CSVs
├── models/
│   ├── finetuned_minilm/   # PyTorch model
│   ├── onnx_model/         # ONNX export
│   └── onnx_quantized/     # Quantized ONNX (113MB)
├── onnx_quantized/         # Local copy for testing
├── r2-model-cdn/           # Cloudflare Worker
├── src/
│   ├── prep_data.py
│   └── train.py
├── index.html
└── readme.md
```

## Troubleshooting

**Labels show as LABEL_0, LABEL_1:**
- Training script sets id2label in model config before saving
- If still wrong, manually edit `models/onnx_quantized/config.json` before upload

**404 errors from CDN:**
- Verify files uploaded with correct prefix: `curl https://your-worker.workers.dev/onnx_quantized/config.json`
- Check R2 bucket in Cloudflare dashboard

**HuggingFace prepending to URL:**
- Use `env.localModelPath` and `env.allowRemoteModels = false` in browser code
- Ensure model path is relative: `pipeline('text-classification', 'onnx_quantized')`
