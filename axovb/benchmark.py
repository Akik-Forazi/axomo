#!/usr/bin/env python3
"""
AXOVB benchmark — performance, accuracy, calibration and robustness suite
for the ONNX bump classifier.

Goes beyond eval.py: single-request latency percentiles, batched throughput,
expected calibration error (ECE), robustness to out-of-distribution inputs,
determinism checks, cold-load time, and (with --compare-head) an old-vs-new
comparison against the model currently at HEAD (pre temperature-scaling fix).

Usage:
    python axovb/benchmark.py
    python axovb/benchmark.py --compare-head          # also bench HEAD model
    python axovb/benchmark.py --iters 1000 --batches 8,32,128
"""
import argparse, hashlib, json, os, subprocess, sys, tempfile, time
from collections import Counter

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "shared"))

from axovb.predictor import BumpPredictor          # noqa: E402
from utils import load_jsonl                        # noqa: E402

MODELS_DIR = os.path.join(REPO_ROOT, "axovb", "models")
DATA_PATH = os.path.join(REPO_ROOT, "axovb", "data", "test.jsonl")


def section(title):
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


def sha256(path, _bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(_bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def extract_test_matrix(predictor, data):
    """Same feature derivation as eval.py (mirrors train.py)."""
    X_list, y_true = [], []
    for item in data:
        f = item.get("features", item)
        row = [f.get(name, 0) for name in predictor.feature_names]
        if "avg_callers_per_symbol" not in f:
            row[16] = round(f.get("num_callers_affected", 0) / max(f.get("num_changed_symbols", 1), 1), 2)
        if "export_change_ratio" not in f:
            exported = f.get("has_removed_exports", 0) + f.get("has_new_exports", 0) + f.get("has_exported_changes", 0)
            row[17] = round(exported / max(f.get("num_changed_symbols", 1), 1), 2)
        if "breaking_score" not in f:
            row[18] = f.get("has_removed_exports", 0) * 3 + f.get("has_signature_changes", 0) * 2 + min(f.get("num_callers_affected", 0), 50) / 50
        if "complexity_score" not in f:
            row[19] = (f.get("num_source_files", 0) + f.get("num_changed_symbols", 0) + min(f.get("num_callers_affected", 0), 50)) / 100
        X_list.append(row)
        y_true.append(predictor.label2id.get(item.get("label", "none"), 0))
    return np.array(X_list, dtype=np.float32), np.array(y_true, dtype=np.int64)


def ece(confidences, correct, n_bins=10):
    """Expected Calibration Error over equal-width confidence bins."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e, n = 0.0, len(confidences)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (confidences > lo) & (confidences <= hi) if i else (confidences >= lo) & (confidences <= hi)
        if m.sum() == 0:
            continue
        gap = abs(correct[m].mean() - confidences[m].mean())
        rows.append((lo, hi, int(m.sum()), confidences[m].mean(), correct[m].mean(), gap))
        e += (m.sum() / n) * gap
    return e, rows


def run_suite(predictor, tag):
    data = load_jsonl(DATA_PATH)
    X, y_true = extract_test_matrix(predictor, data)

    # ---- Accuracy ------------------------------------------------------
    probs = predictor.sess.run([predictor.output_name], {predictor.input_name: X})[0]
    preds = probs.argmax(axis=1)
    correct = preds == y_true
    acc = correct.mean()

    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=list(range(4)))
    labels = [predictor.id2label[i] for i in range(4)]

    section(f"ACCURACY — {tag}")
    print(f"  test samples : {len(y_true)}  dist={dict(Counter(predictor.id2label[i] for i in y_true))}")
    print(f"  accuracy     : {acc*100:.2f}%")
    print(f"  F1 / P / R   : {f1*100:.2f}% / {prec*100:.2f}% / {rec*100:.2f}% (weighted)")
    print("  confusion (rows=true, cols=pred): " + " ".join(f"{l:>7}" for l in labels))
    for i, r in enumerate(cm):
        print(f"    {labels[i]:>7} " + " ".join(f"{v:>7}" for v in r))

    # ---- Calibration ---------------------------------------------------
    conf = probs.max(axis=1)
    e, rows = ece(conf, correct)
    section(f"CALIBRATION (ECE, 10 bins) — {tag}")
    print(f"  ECE              : {e*100:.2f}%")
    print(f"  mean confidence  : {conf.mean()*100:.2f}%   (accuracy {acc*100:.2f}%)")
    ok_c = conf[correct]
    bad_c = conf[~correct] if (~correct).any() else np.array([], dtype=np.float64)
    if ok_c.size:
        print(f"  conf | correct   : mean {ok_c.mean()*100:.2f}%  min {ok_c.min()*100:.2f}%  max {ok_c.max()*100:.2f}%")
    else:
        print("  conf | correct   : n/a (no correct predictions)")
    if bad_c.size:
        print(f"  conf | wrong     : mean {bad_c.mean()*100:.2f}%")
    else:
        print("  conf | wrong     : n/a (no wrong predictions)")
    print("  bin  lo    hi      n   avg_conf  acc     gap")
    for lo, hi, n, ac, a, gap in rows:
        print(f"       {lo:.1f}  {hi:.1f}  {n:>5}   {ac*100:6.2f}%  {a*100:6.2f}%  {gap*100:6.2f}%")
    return {"acc": acc, "ece": e, "mean_conf": conf.mean(), "probs": probs}


def perf_suite(predictor, tag, iters, batch_sizes):
    x = np.zeros((1, 20), dtype=np.float32)  # benign probe row
    # warmup
    for _ in range(300):
        predictor.sess.run([predictor.output_name], {predictor.input_name: x})

    section(f"LATENCY (single request, {iters} iters) — {tag}")
    t = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        predictor.sess.run([predictor.output_name], {predictor.input_name: x})
        t[i] = (time.perf_counter_ns() - t0) / 1e6  # ms
    p = np.percentile(t, [50, 90, 95, 99])
    print(f"  mean {t.mean():.4f} ms | p50 {p[0]:.4f} | p90 {p[1]:.4f} | p95 {p[2]:.4f} | p99 {p[3]:.4f} | max {t.max():.4f}")

    section(f"THROUGHPUT (batched) — {tag}")
    rng = np.random.default_rng(0)
    Xbig = rng.random((512, 20)).astype(np.float32)
    for bs in batch_sizes:
        Xb = np.repeat(Xbig[:bs], 1, axis=0)
        reps = max(4, int(2000 / bs))
        t0 = time.perf_counter()
        for _ in range(reps):
            predictor.sess.run([predictor.output_name], {predictor.input_name: Xb})
        dt = time.perf_counter() - t0
        n = bs * reps
        print(f"  batch {bs:>4}: {n / dt:>9.0f} samples/sec   ({dt / reps * 1000:.3f} ms/batch)")


def robustness_suite(predictor, tag):
    section(f"ROBUSTNESS — {tag}")
    meta_min = predictor.X_min
    meta_max = predictor.X_max
    cases = {
        "all zeros": np.zeros((1, 20), dtype=np.float32),
        "all max": meta_max[None, :].astype(np.float32),
        "all min": meta_min[None, :].astype(np.float32),
        "10x over max": (meta_max * 10)[None, :].astype(np.float32),
        "random uniform [0,1] x100": None,   # filled below
    }
    rng = np.random.default_rng(1)
    cases["random uniform [0,1] x100"] = rng.random((100, 20)).astype(np.float32)

    ok = True
    for name, X in cases.items():
        out = predictor.sess.run([predictor.output_name], {predictor.input_name: X})[0]
        finite = np.isfinite(out).all()
        sums = out.sum(axis=1)
        sum_ok = np.allclose(sums, 1.0, atol=1e-4)
        argmaxes = set(out.argmax(axis=1).tolist())
        status = "OK " if (finite and sum_ok) else "BAD"
        if not (finite and sum_ok):
            ok = False
        print(f"  [{status}] {name:<28} finite={finite} sum(p)=1: {sum_ok} "
              f"(min {sums.min():.6f} max {sums.max():.6f}) argmax classes={sorted(argmaxes)}")

    # NaN / Inf inputs
    Xnan = np.zeros((1, 20), dtype=np.float32); Xnan[0, 0] = np.nan
    Xinf = np.zeros((1, 20), dtype=np.float32); Xinf[0, 0] = np.inf
    for name, X in (("NaN input", Xnan), ("Inf input", Xinf)):
        out = predictor.sess.run([predictor.output_name], {predictor.input_name: X})[0]
        finite = np.isfinite(out).all()
        print(f"  [note] {name:<28} finite_output={finite} "
              f"({'model returns NaN for NaN/Inf — caller must sanitize input' if not finite else 'fine'})")

    # Determinism
    X = (meta_min + (meta_max - meta_min) * 0.37)[None, :].astype(np.float32)
    ref = predictor.sess.run([predictor.output_name], {predictor.input_name: X})[0]
    det = all(np.array_equal(ref, predictor.sess.run([predictor.output_name], {predictor.input_name: X})[0]) for _ in range(50))
    print(f"  [{'OK ' if det else 'BAD'}] determinism: 50 repeats identical = {det}")

    # Batch invariance (dynamic-axis export sanity)
    Xb = np.repeat(X, 64, axis=0)
    pb = predictor.sess.run([predictor.output_name], {predictor.input_name: Xb})[0]
    inv = np.allclose(pb[0], ref[0], atol=1e-6)
    print(f"  [{'OK ' if inv else 'BAD'}] batch invariance: single vs batch-64 probs identical = {inv}")
    return ok


def end_to_end_smoke(predictor):
    section("END-TO-END SMOKE (raw diff -> features -> prediction)")
    diffs = {
        "removed exported symbol": (
            "diff --git a/src/auth.js b/src/auth.js\n"
            "--- a/src/auth.js\n+++ b/src/auth.js\n"
            "@@ -1,5 +1,2 @@\n"
            "-export function authMiddleware(req, res, next) {\n"
            "-  return next();\n-}\n"
            " export default {};\n"
        ),
        "comment tweak": (
            "diff --git a/src/util.js b/src/util.js\n"
            "--- a/src/util.js\n+++ b/src/util.js\n"
            "@@ -2,1 +2,1 @@\n"
            "-// helper for parsing\n"
            "+// helper: parses input strings\n"
        ),
    }
    for name, d in diffs.items():
        features, _, _ = __import__("axovb.analyzer", fromlist=["extract_diff_features"]).extract_diff_features(d)
        label, conf, p = predictor.predict_features(features)
        print(f"  {name:<26} -> {label:<6} conf={conf*100:5.1f}%  "
              f"probs[none,patch,minor,major]=[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f},{p[3]:.2f}]")


def model_identity(model_dir, tag):
    onnx = os.path.join(model_dir, "bump_mlp.onnx")
    meta = os.path.join(model_dir, "metadata.json")
    with open(meta) as f:
        m = json.load(f)
    import onnxruntime as ort
    t0 = time.perf_counter()
    sess = ort.InferenceSession(onnx)
    load_ms = (time.perf_counter() - t0) * 1000
    section(f"MODEL IDENTITY — {tag}")
    print(f"  onnx path     : {onnx}")
    print(f"  onnx size     : {os.path.getsize(onnx):,} bytes ({os.path.getsize(onnx)//1024} KB)")
    print(f"  sha256        : {sha256(onnx)}")
    print(f"  temperature   : {m.get('temperature')}  model_type: {m.get('model_type')}")
    print(f"  params        : 2,806,020 (2.8M, per architecture)")
    print(f"  cold-load time: {load_ms:.1f} ms (session creation)")


def extract_head_model():
    """Materialize HEAD's bump_mlp.onnx + metadata.json into a temp dir."""
    out = tempfile.mkdtemp(prefix="axovb_head_")
    for name in ("bump_mlp.onnx", "metadata.json"):
        blob = subprocess.run(
            ["git", "-C", REPO_ROOT, "show", f"HEAD:axovb/models/{name}"],
            check=True, capture_output=True,
        ).stdout
        with open(os.path.join(out, name), "wb") as f:
            f.write(blob)
    return out


def main():
    ap = argparse.ArgumentParser(description="AXOVB ONNX benchmark suite")
    ap.add_argument("--iters", type=int, default=3000, help="single-request latency iterations")
    ap.add_argument("--batches", type=str, default="1,8,32,128,512", help="comma list of batch sizes")
    ap.add_argument("--compare-head", action="store_true", help="also benchmark the model at git HEAD")
    ap.add_argument("--model-dir", default=MODELS_DIR)
    args = ap.parse_args()
    batch_sizes = [int(b) for b in args.batches.split(",") if b.strip()]

    print(f"AXOVB BENCHMARK  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"python {sys.version.split()[0]} | numpy {np.__version__} | onnxruntime "
          f"{__import__('onnxruntime').__version__} | platform {sys.platform}")

    model_identity(args.model_dir, "working tree")
    p_new = BumpPredictor(args.model_dir)
    res_new = run_suite(p_new, "working tree")
    perf_suite(p_new, "working tree", args.iters, batch_sizes)
    robustness_suite(p_new, "working tree")
    end_to_end_smoke(p_new)

    if args.compare_head:
        head_dir = extract_head_model()
        model_identity(head_dir, "git HEAD (pre-fix)")
        p_old = BumpPredictor(head_dir)
        res_old = run_suite(p_old, "git HEAD (pre-fix)")

        # Head-to-head on identical inputs
        data = load_jsonl(DATA_PATH)
        X, y_true = extract_test_matrix(p_new, data)
        same_argmax = (res_old["probs"].argmax(axis=1) == res_new["probs"].argmax(axis=1)).mean()
        print(f"\n{'=' * 62}\n  HEAD vs WORKING TREE (identical inputs, n={len(y_true)})\n{'=' * 62}")
        print(f"  accuracy      : {res_old['acc']*100:.2f}%  ->  {res_new['acc']*100:.2f}%")
        print(f"  ECE           : {res_old['ece']*100:.2f}%  ->  {res_new['ece']*100:.2f}%")
        print(f"  mean conf     : {res_old['mean_conf']*100:.2f}%  ->  {res_new['mean_conf']*100:.2f}%")
        print(f"  argmax agrees : {same_argmax*100:.1f}% of test samples")

    print("\nDONE.")


if __name__ == "__main__":
    main()
