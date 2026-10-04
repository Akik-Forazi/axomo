# AXOMO — Axo Models

**Fine-tuned sub-200M parameter ONNX models for the Fraziym ecosystem.**

V00.01.000-beta-01 · FRAZIYM versioning

## What is AXOMO?

AXOMO contains the training scripts, evaluation tools, and exported ONNX models for 4 specialized small models used by AXOVB and AXOTEST. Each model is purpose-built for ONE task — not a general-purpose chatbot.

## Models

| Model | For | Base | Params | Task | Input → Output |
|---|---|---|---|---|---|
| `axovb-bump-classifier` | AXOVB | DistilBERT | 66M | text-classification | git diff → {patch\|minor\|major\|none} |
| `axotest-unit-generator` | AXOTEST Unit | CodeT5-small | 60M | text2text-generation | source code → test code |
| `axotest-security-scanner` | AXOTEST Security | DistilBERT | 66M | text-classification | code → {secure\|vulnerable} |
| `axotest-edge-detector` | AXOTEST Edge | DistilBERT | 66M | token-classification | code → boundary tokens |

All models run locally on CPU via `@huggingface/transformers` (ONNX Runtime). No API calls. Inference: 10-50ms per request.

## Structure

```
axomo/
├── axovb/                    # AXOVB bump classifier
│   ├── prepare_data.py       # Scrape npm publish history for (diff → bump) pairs
│   ├── train.py              # Fine-tune DistilBERT (66M)
│   ├── eval.py               # Evaluate accuracy + inference speed
│   └── export_onnx.py → ../shared/export_onnx.py
├── axotest-unit/             # AXOTEST unit test generator
│   ├── prepare_data.py       # Collect (source, test) pairs from GitHub
│   ├── train.py              # Fine-tune CodeT5-small (60M)
│   └── eval.py               # Generate tests, run with vitest, measure pass rate
├── axotest-security/         # AXOTEST security scanner
│   ├── prepare_data.py       # Collect (code, vulnerable?) pairs
│   ├── train.py              # Fine-tune DistilBERT (66M)
│   └── eval.py               # Accuracy + F1 + inference speed
├── axotest-edge/             # AXOTEST edge case detector
│   ├── prepare_data.py       # Collect code with labeled boundary tokens
│   ├── train.py              # Fine-tune DistilBERT (66M) for token classification
│   └── eval.py               # Token-level F1 + entity precision/recall
├── shared/
│   ├── utils.py              # Shared data utilities
│   └── export_onnx.py        # Export any model to ONNX format
├── requirements.txt          # Python deps (torch, transformers, optimum, onnx)
└── package.json              # npm package metadata (@fraziym/axomo)
```

## Training workflow

```bash
pip install -r requirements.txt

cd axovb/
python prepare_data.py --output ./data/
python train.py --data ./data/train.jsonl --output ./model/
python eval.py --model ./model/ --data ./data/test.jsonl
python ../shared/export_onnx.py --model ./model/ --output ./onnx/ --task text-classification
```

## Integration

After training + exporting to ONNX, copy the model to AXOVB/AXOTEST:

```bash
# AXOVB
cp -r axovb/onnx/ ~/.axovb/models/bump-classifier/
axovb check --model-path ~/.axovb/models/bump-classifier/

# AXOTEST
cp -r axotest-unit/onnx/ ~/.axotest/models/unit-generator/
axotest run --model-path ~/.axotest/models/unit-generator/
```

## Attribution

The fine-tuning scripts use [HuggingFace Transformers](https://huggingface.co/docs/transformers) and [ONNX Runtime](https://onnxruntime.ai/).

Base models from the HuggingFace Hub:
- [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) — 66M params
- [Salesforce/codet5-small](https://huggingface.co/Salesforce/codet5-small) — 60M params

## License

MIT © 2026 Akik Faraji — Fraziym Tech & AI
