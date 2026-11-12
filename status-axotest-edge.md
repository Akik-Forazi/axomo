# axotest-edge/ 状态调查（2026-10-18）

> 目标路径 .zwork/status-axotest-edge.md 被运行时保留、禁止写入，故落在此处（status-axotest-edge.md）。

## 结论
数据准备已完成（1005 train + 178 test），但 prep 脚本在最后统计打印步崩溃（数据无损）；训练尚未开始；下一步是运行 train.py，之后 eval.py 与 ONNX 导出。

## 1. 目录清单

```
axotest-edge/
├── prepare_data.py   tracked（数据采集脚本）
├── train.py          tracked（DistilBERT 66M，token 分类，3 标签 O/B-BOUNDARY/I-BOUNDARY）
├── eval.py           tracked（HF pipeline 评测，weighted F1/P/R + 延迟）
├── prep.log          untracked，25 行
└── data/             untracked
    ├── train.jsonl   2,734,546 bytes，1005 行
    ├── test.jsonl      516,547 bytes， 178 行
    └── _clones/      lodash、ramda、yup、zod 四个浅克隆（depth=50），
                       递归文件合计 23,459,566 bytes（~23.5 MB）
```

- git status：`?? axotest-edge/data/`、`?? axotest-edge/prep.log`，与任务描述一致。
- `_clones/` 已被 .gitignore 规则 `axotest-*/data/_clones/` 忽略（.gitignore 近期改动仅新增 `.env`，未动这条规则）。
- train.py / eval.py / prepare_data.py 最后一次改动：commit 3475561 "fix: transformers 5.x compat + edge pipeline + export_onnx rewrite"（当前 HEAD = e2ef5d8）。

## 2. prep.log 分析（全文即 25 行）

日志时间线：
1. `Scraping lodash... / ramda... / yup... / zod...` —— 四仓库克隆与扫描全部执行。
2. `Dataset statistics: Train: 1005 samples, Test: 178 samples`。
3. Traceback：

```
File "axotest-edge/prepare_data.py", line 75, in main
    print_stats(train, test, "labels")
File "axotest-edge/../shared/utils.py", line 48, in print_stats
    labels = Counter(item.get(label_field, "unknown") for item in train)
TypeError: unhashable type: 'list'
```

**崩溃点与数据完整性判定（一手代码 + 磁盘证据）：**
- prepare_data.py main() 的执行顺序（:72-76）：`train_test_split` → `save_jsonl(train)` → `save_jsonl(test)` → `print_stats(...)` → `print("Data saved to ...")`。
- 磁盘上 train.jsonl = 1005 行、test.jsonl = 178 行，与日志统计完全一致 ⇒ 两份 jsonl 在崩溃前已写入成功。
- 崩溃原因：token 分类样本的 `labels` 字段是 list（BIO 序列），`Counter` 不能哈希 list。纯统计打印 bug，发生在保存之后，"Data saved to ..." 因异常未打印。
- **判定：数据准备实质完成；脚本以异常退出，但产物完好。**

样本格式（train.jsonl 首行，ramda `_isPlaceholder.js`）：`{"tokens": [...], "labels": [...], "file": "..."}`，whitespace 分词 + BIO（prepare_data.py:18-24 的 label_tokens 实际只产出 B-BOUNDARY/O）。

## 3. 训练与导出状态

- `Test-Path model / onnx / checkpoints` → 全部 False；axotest-edge/ 下除 data/ 无任何其他目录 ⇒ **训练从未启动，无任何模型或 ONNX 产物**。
- train.py（:15-19 参数）：`--data --output --base-model(distilbert-base-uncased) --epochs 5 --batch-size 16 --lr 2e-5`；内部自带 train_test_split 再切分；Trainer 按 epoch 保存/评估，`trainer.save_model(args.output)` + tokenizer 保存（train.py 尾部）。
- train.py 末尾打印导出提示：`python ../shared/export_onnx.py --model {output} --output {output}-onnx/ --task token-classification`。
- eval.py：加载 `--model`（HF pipeline, CPU）对 `--data` 逐样本推理，输出 weighted F1/P/R。

## 4. 下一步管线（按序）
1. （可选，1 行修复）shared/utils.py:48 对 list 型标签先展平或跳过，否则 prep/统计类脚本仍会在收尾崩溃（axotest-security、axotest-unit 的 prep.log 同为 untracked，疑似同一 bug 受害者，待查）。
2. `python axotest-edge/train.py --data axotest-edge/data/train.jsonl --output axotest-edge/model`（产出 model/，含 checkpoint-*）。
3. `python axotest-edge/eval.py --model axotest-edge/model --data axotest-edge/data/test.jsonl`。
4. `python shared/export_onnx.py --model axotest-edge/model --output axotest-edge/model-onnx/ --task token-classification`，产物进 onnx/（与 axovb/semver-v1-c0 的既有模式对齐）。

## 证据等级
- 一手代码：axotest-edge/{prepare_data,train,eval}.py、shared/utils.py:48、.gitignore、git status/log —— 最高等级。
- 一手产物：data/*.jsonl 行数与字节数、prep.log 全文、_clones/ 尺寸（实测）。
- 推断（已标注）：security/unit 两模块受同一 print_stats bug 影响 —— 仅基于同构目录布局，未读其日志，属待证。
