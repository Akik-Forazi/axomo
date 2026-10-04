#!/usr/bin/env python3
"""
Shared utilities for model training.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], path: str):
    """Save data as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def train_test_split(data: List[Dict], test_ratio: float = 0.15):
    """Simple train/test split."""
    import random
    random.seed(42)
    shuffled = data.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split], shuffled[split:]

def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate text to fit model's max input length."""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text

def print_stats(train, test, label_field: str = "label"):
    """Print dataset statistics."""
    from collections import Counter
    print(f"\nDataset statistics:")
    print(f"  Train: {len(train)} samples")
    print(f"  Test:  {len(test)} samples")
    if train:
        labels = Counter(item.get(label_field, "unknown") for item in train)
        print(f"  Label distribution (train):")
        for label, count in labels.most_common():
            print(f"    {label}: {count} ({count/len(train)*100:.1f}%)")
    print()
