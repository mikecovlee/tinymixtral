# TinyMixtral — 小型 MoE 因果 LM 预训练项目

## 硬件
- CPU: AMD Ryzen 7 5800X
- GPU: NVIDIA RTX A5000 24GB
- RAM: 32GB

## 环境
```bash
conda create -n tinymixtral python=3.13 -y
pip install torch transformers datasets sentencepiece scikit-learn
```

代理（如需）：`export http_proxy=http://127.0.0.1:7890/ https_proxy=http://127.0.0.1:7890/`

## 当前模型：896d/10L/6E (~432M)

| 参数 | 值 |
|------|-----|
| hidden_size | 896 |
| num_layers | 10 |
| num_experts | 6 (top-2) |
| GQA | 14 heads / 2 KV heads |
| expert_intermediate | 2389 (8/3×) |
| vocab_size | 32000 |
| total params | ~432M |
| active params | ~175M |

## 快速命令

```bash
# 1. 下载 tokenizer
python scripts/prepare_tokenizer.py --from-hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output tokenizer/

# 2. Tokenize 数据
python scripts/prepare_data.py --dataset allenai/c4 --subset en \
  --tokenizer tokenizer/ --output data/c4/tokenized --max-tokens 4000000000 --force

# 3. 预训练
python scripts/train.py --cache-dir data/c4/tokenized \
  --batch-size 22 --max-tokens 4000000000 \
  --keep-last-checkpoints 5

# 4. 从 checkpoint 恢复
python scripts/resume.py --batch-size 22

# 5. GLUE 评测
python scripts/eval_glue.py --checkpoint checkpoints/run/step_0005000 \
  --tokenizer tokenizer/ --tasks quick --limit 200

# 6. 全任务评测
python scripts/eval_glue.py --checkpoint checkpoints/run/step_0005000 \
  --tokenizer tokenizer/ --tasks all --limit 500 --output evals/result.json
```

`prepare_data.py` 会为每篇文档显式追加 EOS，并强制 tokenizer 词表与模型的 32K 配置一致。必须在 `--max-tokens` 与 `--max-samples` 中选择一个停止目标；4B 训练建议使用 `--max-tokens 4000000000`，数据源提前结束时不会发布不足 4B 的半成品。输出目录非空时需加 `--force` 完整重建。新 shard 先写入临时目录，再通过 backup/rename 替换；若替换中断，下次运行会恢复唯一备份并识别跨进程残留目录。

## 项目结构
```
├── model/                 # 模型代码
│   ├── config.py          # TinyMixtralConfig
│   └── modeling.py        # MoE+GQA+RoPE+RMSNorm
├── evals/                 # GLUE 评测
│   ├── prompt_scoring.py  # Zero-shot log-likelihood 评分
│   ├── glue_tasks.py      # 7 GLUE 任务模板+verbalizer
│   └── metrics.py         # accuracy, F1, MCC
├── scripts/
│   ├── train.py           # 预训练
│   ├── resume.py          # checkpoint 恢复
│   ├── eval_glue.py       # GLUE CLI
│   ├── chat.py            # 交互式生成
│   ├── prepare_data.py    # 数据 pre-tokenize
│   ├── prepare_tokenizer.py
│   ├── benchmark.py       # 硬件适配
│   └── search.py          # 超参搜索
├── tokenizer/             # TinyLlama tokenizer (vocab=32000)
├── checkpoints/run/       # 模型检查点 + training_state.pt
└── data/c4/tokenized/     # 预 tokenize 的 .pt shard
```

## 关键设计

1. **数据预 tokenize** — `prepare_data.py` 流式读取数据，增量写入本地 `.pt` shard，并为每篇文档追加 EOS。
2. **分片加载** — 训练时逐个 shard 循环，避免全量加载 OOM。
3. **纯 BF16 训练** — 模型参数及 AdamW 一、二阶动量均使用 BF16；矩阵权重使用 weight decay，RMSNorm 等 1D 参数不衰减。
4. **Activation checkpointing** — 必须开启，否则 OOM。
5. **Aux loss 梯度** — `f_i` detach（离散选择），`P_i` 保留梯度（softmax 可微）；各层 aux 按层平均后应用 0.01 系数，量级不随层数放大。
6. **完整 checkpoint** — `training_state.pt` 保存 optimizer、scheduler、step、token 计数、schedule 参数、`batch_size/seq_len` 及数据游标 `fi/ptr`；checkpoint 完整写入临时目录后再原子发布。
7. **长训保护** — 启动时预检 checkpoint 磁盘空间，默认仅保留最近 5 个周期 checkpoint（final 永不清理）；SIGINT/SIGTERM 会在当前 step 完成后保存再退出。

> 训练停止按 `--max-tokens` / `--max-steps` 确定。

> checkpoint 保存后默认不运行评测；需要同步执行 CPU GLUE 时添加 `--eval-on-save`。
