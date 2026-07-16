---
name: nvidia-nlp-finetune-llama-7b
description: NVIDIA GPU 上 Alpaca-LoRA/LLaMA-7B 语言模型微调性能评测技能。基于 alpaca-lora 项目和 scripts/batch_finetune.sh，指导 executor 启动容器、执行微调、采集训练日志，并生成包含 train_tokens_per_second、单卡吞吐、运行时间和 loss 的 result.json。用于 NVIDIA 多卡 LoRA 微调、Alpaca 微调性能测试及 TGS 指标采集。
test_case: llama-7B
---

# nvidia-nlp-finetune: llama-7B

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑语言模型微调"
- "帮我测试 LoRA 微调性能"
- "我要跑 Alpaca-LoRA 微调"
- "用 LLaMA-7B 跑 alpaca-lora finetune"
- "采集 alpaca-lora 训练 train_tokens_per_second(tgs)"

## 硬件要求

- 8 张 NVIDIA GPU

## 依赖要求

**Docker 镜像**：
```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-finetune:latest
```

## 环境变量

### 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `MODEL_DIR` | `/data/models/llama-7b-hf` | 是 | 模型权重目录，存放 LLaMA-7B 权重（如 yahma/llama-7b-hf） |
| `DATASET_DIR` | `/data/datasets/alpaca-cleaned` | 是 | 微调数据集目录，存放 Alpaca-cleaned 指令微调数据 |
| `CODE_DIR` | `/workspace/code/alpaca_finetune` | 是 | 微调代码目录，包含训练脚本、配置文件与输出子目录 |
| `RESULTS_DIR` | `/workspace/results` | 是 | 评测结果目录，存放 metrics 汇总文件 `result.json` |
| `LOGS_DIR` | `/workspace/logs` | 是 | 日志目录，存放训练日志（`finetune_<bs>_<mbs>_closeint8.log`）、执行过程输出与内部报错信息 |

**说明**：
- **MODEL_DIR** 需要外部提供，挂载预训练模型权重目录（HuggingFace 格式）
- **DATASET_DIR** 需要外部提供，挂载微调数据集目录
- **CODE_DIR** 需要外部提供，挂载 alpaca_finetune 训练代码目录。`finetune.sh` 在 `$CODE_DIR/alpaca-lora/` 下执行：LoRA 权重输出到 `$CODE_DIR/alpaca-lora/lora-adapter/`；训练日志由 `finetune.sh` 直接写入 `$LOGS_DIR/finetune_128_4_closeint8.log`（batch size = 128，micro batch size = 4）
- **RESULTS_DIR** 需要外部提供，挂载评测结果目录。所有结构化产物（metrics、状态汇总）以 `result.json` 形式写入此目录,供上层 agent 拉取与展示
- **LOGS_DIR** 需要外部提供，挂载日志目录。训练日志、`stdout`/`stderr` 重定向、容器内异常堆栈等运行期文本均写入此目录，便于事后排查
- 表格中的"映射目录"列指明了容器启动时 `-v` 参数的挂载路径，即宿主机路径映射到容器内的路径

### 批量脚本变量

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `NLP_FIN_PROJECT_ROOT` | `/workspace/code/alpaca_finetune` | 微调项目根目录 |
| `NLP_FIN_ALPACA_DIR` | `$NLP_FIN_PROJECT_ROOT/alpaca-lora` | `finetune_int8_close.py` 所在目录 |
| `NLP_FIN_LOGS_DIR` | `/workspace/logs` | 日志目录 |
| `NLP_FIN_RESULTS_DIR` | `/workspace/results` | 结构化结果目录 |
| `NLP_FIN_NGPU` | `${CARD_COUNT:-8}` | `torch.distributed.run --nproc_per_node` 的值，同时用于计算单卡吞吐 |
| `NLP_FIN_BATCH_SIZE` | `128` | 训练 batch size，也用于日志文件名 |
| `NLP_FIN_MICRO_BATCH_SIZE` | `4` | 训练 micro batch size，也用于日志文件名 |
| `NLP_FIN_LOG_FILE` | `$NLP_FIN_LOGS_DIR/finetune_<bs>_<mbs>_closeint8.log` | 训练日志完整路径 |
| `NLP_FIN_RESULT_FILE` | `$NLP_FIN_RESULTS_DIR/result.json` | 结果文件完整路径 |
| `PYTHON` | `python3` | 训练用 python 可执行文件，容器 PATH 内需可用 |
| `MODEL_DIR`（容器内） | `/data/models/llama-7b-hf` | `--base_model` 指向的模型目录（与宿主机挂载变量同名，这里是容器内解析值） |
| `DATASET_DIR`（容器内） | `/data/datasets/alpaca-cleaned` | `--data_path` 指向的数据集目录 |
| `OUTPUT_DIR` | `$NLP_FIN_ALPACA_DIR/lora-adapter` | LoRA adapter 输出目录 |

`batch_finetune.sh` 已把 `finetune.sh` 的训练逻辑（`torch.distributed.run finetune_int8_close.py`）合并进来，不再依赖项目自带的 `finetune.sh`。所有训练参数（`--base_model`、`--data_path`、`--num_epochs` 等）由 `batch_finetune.sh` 直接传给 `finetune_int8_close.py`，默认值用上表容器内挂载点。切换 batch size、micro batch size 或日志命名时，设置相应变量或直接设置 `NLP_FIN_LOG_FILE`。

**目录结构说明**：

- `$MODEL_DIR`: 模型权重目录，采用 HuggingFace Hub 缓存布局，典型结构如下：
  ```
  $MODEL_DIR/                    # 例如 models--yahma--llama-7b-hf
  ├── blobs/                     # 实际权重文件（哈希命名）
  ├── refs/                      # 分支/标签引用
  └── snapshots/                 # 各 commit 快照（软链至 blobs/）
      └── <commit_hash>/         # 例如 cf33055e5df9cc533abd7ea4707bf727ca2ada75
          ├── config.json
          ├── generation_config.json
          ├── tokenizer.model
          ├── tokenizer_config.json
          ├── special_tokens_map.json
          ├── pytorch_model-00001-of-00002.bin
          ├── pytorch_model-00002-of-00002.bin
          └── pytorch_model.bin.index.json
  ```

  **注意**：使用时需指向具体的 snapshot 子目录（即 `$MODEL_DIR/snapshots/<commit_hash>`），而非 `$MODEL_DIR` 本身。`snapshots/` 下可能存在多个 commit 版本，按需选择。

- `$DATASET_DIR`: 数据集目录，典型结构如下：
  ```
  $DATASET_DIR/
  └── alpaca_data_cleaned.json   # Alpaca 指令微调数据
  ```

- `$CODE_DIR`: 微调代码目录，典型结构如下：
  ```
  $CODE_DIR/                          # alpaca_finetune
  ├── alpaca-lora/                    # finetune.sh 的工作目录
  │   ├── finetune.sh                 # 启动脚本（torchrun --nproc_per_node=8 入口）
  │   ├── finetune_int8_close.py      # 主训练脚本（关闭 int8 量化）
  │   ├── utils/
  │   │   └── prompter.py             # 被 finetune_int8_close.py 导入：from utils.prompter import Prompter
  │   ├── templates/                  # Prompter 加载的 prompt 模板
  │   │   └── alpaca.json             # 默认模板（prompt_template_name="alpaca"）
  │   └── lora-adapter/               # LoRA 权重输出目录（脚本参数 --output_dir './lora-adapter'）
  └── peft/                           # PEFT 库源码（finetune_int8_close.py 导入：from peft import ...）
  ```

  **注意**：
  - `finetune.sh`（项目自带的训练启动脚本）**不再需要**——`batch_finetune.sh` 直接调用 `finetune_int8_close.py`，所有路径和参数由 `batch_finetune.sh` 用容器内挂载点默认值提供。
  - `batch_finetune.sh` 在 `$CODE_DIR/alpaca-lora/` 下执行，`--output_dir './lora-adapter'` 与日志重定向 `> "$LOG_FILE"` 均使用相对路径，产物落在该目录内
`trainer.py` 和 `trainer_utils.py` 是项目自带的 transformers 补丁版本（见 `$CODE_DIR/README.md`），在训练前由 `batch_finetune.sh` 自动覆盖到容器内 `import transformers` 解析出的 site-packages 目录。不覆盖的话，汇总行没有 `train_tokens_per_second` 字段，解析会失败。使用 HuggingFace Hub 缓存布局时，把 `MODEL_DIR` 指向包含 `snapshots/<commit_hash>/` 的缓存根；扁平布局则直接指向含 `config.json` 的目录。


镜像内 `python3` 在 PATH 中（`/opt/conda/bin/python3` 或 `/usr/bin/python3`），`torch.distributed.run` 可用。

- `$RESULTS_DIR`: 评测结果目录，典型结构如下：
  ```
  $RESULTS_DIR/
  └── result.json   # 指标采集脚本生成的结构化结果（{"status": "success", "metrics": {...}}）
  ```

  **注意**：内容由步骤 3 的指标采集脚本写入；上层 agent 会从该路径（容器内 `/workspace/results/result.json`）读取或从脚本 stdout 解析 metrics。

- `$LOGS_DIR`: 日志目录，典型结构如下：
  ```
  $LOGS_DIR/
  └── finetune_128_4_closeint8.log   # 训练日志（finetune.sh 重定向产物）
  ```

  **注意**：`batch_finetune.sh` 内 `$LOG_FILE` 已直接指向指定路径，训练日志直接落到该路径。步骤 3 中 `log_path = '/workspace/logs/finetune_128_4_closeint8.log'` 即为该文件的容器内路径。

**注意**：
- 必需的参数（`MODEL_DIR`、`DATASET_DIR`、`CODE_DIR`、`RESULTS_DIR`、`LOGS_DIR`）必须提供
- 容器内路径已通过卷挂载固定，对应 `docker run` 命令中的 `-v` 参数
- 宿主机路径建议存放在大容量磁盘上，避免占用系统盘空间

## 执行流程

### 步骤 1：容器启动

**挂载权限约定**：
- `:ro` — 只读，用于输入数据（模型权重、数据集等），防止误修改
- `:rw` — 读写，用于输出目录（代码目录下的 LoRA 权重、训练日志等）

**完整启动命令**：

```bash
docker run -it \
  --name alpaca_finetune \
  --gpus all \
  --shm-size=128g \
  -v $MODEL_DIR:/data/models/llama-7b-hf:ro \
  -v $DATASET_DIR:/data/datasets/alpaca-cleaned:ro \
  -v $CODE_DIR:/workspace/code/alpaca_finetune:rw \
  -v $RESULTS_DIR:/workspace/results:rw \
  -v $LOGS_DIR:/workspace/logs:rw \
  swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-finetune:latest \
  bash
```

**注意**：
- 所有大文件路径通过 `MODEL_DIR`、`DATASET_DIR`、`CODE_DIR` 环境变量提供，避免命令中硬编码
- 若已存在同名容器，先执行 `docker rm -f alpaca_finetune`
- `CODE_DIR` 必须挂载为 `:rw`，因为 LoRA 权重与训练日志会写入 `$CODE_DIR/alpaca-lora/` 子目录

#### 容器管理命令

**进入已创建的容器**：
```bash
# 如果容器已在运行
docker exec -it alpaca_finetune /bin/bash

# 如果容器已停止，先启动再进入
docker start alpaca_finetune
docker exec -it alpaca_finetune /bin/bash
```

**验证容器环境**：
```bash
# 检查 GPU 设备
nvidia-smi

# 检查挂载的目录
ls -lh /data/models/llama-7b-hf/
ls -lh /data/datasets/alpaca-cleaned/
ls -lh /workspace/code/alpaca_finetune/

# 检查 alpaca-lora 工作目录
ls -lh /workspace/code/alpaca_finetune/alpaca-lora/
```

### 步骤 2：执行评测

优先使用本 skill 提供的 `scripts/batch_finetune.sh`。如果评测框架将 skill 脚本放在 `/workspace/scripts`，复制脚本后执行：

```bash
if [ ! -f /workspace/code/alpaca_finetune/batch_finetune.sh ] && \
   [ -f /workspace/scripts/batch_finetune.sh ]; then
  cp /workspace/scripts/batch_finetune.sh /workspace/code/alpaca_finetune/batch_finetune.sh
  chmod +x /workspace/code/alpaca_finetune/batch_finetune.sh
fi
test -f /workspace/code/alpaca_finetune/batch_finetune.sh

export NLP_FIN_NGPU="${CARD_COUNT:-8}"
bash /workspace/code/alpaca_finetune/batch_finetune.sh
```

脚本按顺序完成以下操作：

1. 校验 GPU 数、项目目录和 `finetune_int8_close.py`。
2. 创建日志、结果、LoRA adapter 输出目录，并记录本次运行时间标记。
3. 用 `$CODE_DIR/trainer.py` 和 `$CODE_DIR/trainer_utils.py` 覆盖容器内 `transformers` site-packages 里的同名文件（`import transformers` 解析出路径），让 Trainer 汇总行带 `train_tokens_per_second` 等性能字段。
4. 在 `alpaca-lora/` 中执行 `python -m torch.distributed.run --nproc_per_node=$NLP_FIN_NGPU finetune_int8_close.py`，参数（`--base_model`/`--data_path`/`--output_dir` 等）由脚本用容器内挂载点默认值提供。
5. 拒绝缺失或未被本次运行更新的日志，避免读取历史结果。
6. 提取最后一个包含 `train_tokens_per_second` 的训练汇总 dict。
7. 计算单卡吞吐并写入 `/workspace/results/result.json`。
8. 以 `result.json: {...}` 格式回显结果，供上层 agent 解析。

若代码目录不可写，直接从预置路径执行脚本，并显式指定项目根目录：

```bash
export NLP_FIN_PROJECT_ROOT=/workspace/code/alpaca_finetune
export NLP_FIN_NGPU="${CARD_COUNT:-8}"
bash /workspace/scripts/batch_finetune.sh
```

**输出产物**：

| 文件 / 目录 | 容器内路径 | 描述 |
| :--- | :--- | :--- |
| `lora-adapter/` | `/workspace/code/alpaca_finetune/alpaca-lora/lora-adapter/` | LoRA 权重输出目录 |
| `finetune_<bs>_<mbs>_closeint8.log` | `/workspace/logs/finetune_128_4_closeint8.log` | 训练日志，由 `finetune.sh` 内 `> "$LOG_FILE"` 重定向生成 |

**验证训练结果**：
```bash
# 检查 LoRA 权重输出
ls -lh /workspace/code/alpaca_finetune/alpaca-lora/lora-adapter/

# 查看训练日志末尾
tail -50 /workspace/logs/finetune_128_4_closeint8.log
```

**注意**：
- 必须先 `cd` 到 `alpaca-lora/` 目录再执行，否则 `--output_dir './lora-adapter'` 与日志重定向的相对路径会错位
- 如果切换了 batch size / micro batch size，日志文件名会变（格式 `finetune_<bs>_<mbs>_closeint8.log`），步骤 3 中的 `log_path` 需要同步更新

### 步骤 3：指标采集

训练完成后，`finetune_128_4_closeint8.log` 末尾会出现一行 Python dict 形式的汇总信息（由 `transformers.Trainer` 在训练结束时打印），所有性能指标均从该行提取：

```
{'train_runtime': 0, 'train_samples_per_second': 0, 'train_steps_per_second': 0, 'train_tokens_per_second': 0, 'train_loss': 0, 'epoch': 0}
```

**注意区分**：训练过程中频繁出现的 `{'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}` 是单步 loss 日志，**不是**最终汇总行；只有包含 `train_runtime` / `train_tokens_per_second` 字段的那一行才是。

#### 关键性能指标

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `train_tokens_per_second` | 训练总吞吐（tokens/s，全局聚合） |
| 性能（必采） | `tokens_per_sec_per_gpu` | 单卡吞吐 = `train_tokens_per_second / nproc_per_node`（默认 8） |
| 性能（辅助） | `train_samples_per_second` | 每秒处理样本数 |
| 性能（辅助） | `train_steps_per_second` | 每秒迭代步数 |
| 性能（辅助） | `train_runtime` | 总训练耗时（秒） |
| 质量（辅助） | `train_loss` | 训练最终 loss（越低越好） |

#### 指标采集方法

**Python 脚本提取**

脚本职责：
1. 从训练日志中提取最后一次包含 `train_tokens_per_second` 的汇总 dict
2. 计算 `tokens_per_sec_per_gpu`（总吞吐 / nproc）
3. 把 metrics 写入 `/workspace/results/result.json`（`{"status": "success", "metrics": {...}}` 格式）
4. 同时把 `result.json` 的内容回显到 stdout（前缀 `result.json: `），供 agent 从标准输出解析

```bash
python - <<'EOF'
import ast
import json
import os
import re

log_path = '/workspace/logs/finetune_128_4_closeint8.log'
result_path = '/workspace/results/result.json'

with open(log_path) as f:
    text = f.read()

# 抓取所有包含 train_tokens_per_second 的 dict 行，取最后一次
matches = re.findall(r"\{[^{}]*'train_tokens_per_second':[^{}]*\}", text)
if not matches:
    raise SystemExit("未找到训练汇总行，训练可能未结束或日志被截断")

summary = ast.literal_eval(matches[-1])
nproc = 8  # 与 finetune.sh 中的 --nproc_per_node 保持一致
ttps  = summary['train_tokens_per_second']

metrics = {
    'train_tokens_per_second': round(ttps, 2),
    'tokens_per_sec_per_gpu':  round(ttps / nproc, 2),
    'train_samples_per_second': round(summary['train_samples_per_second'], 2),
    'train_steps_per_second':   round(summary['train_steps_per_second'], 4),
    'train_runtime':            round(summary['train_runtime'], 2),
    'train_loss':               round(summary['train_loss'], 4),
}

# 1) 控制台人类可读打印
print(f"train_tokens_per_second (total)   : {metrics['train_tokens_per_second']:.2f}")
print(f"tokens_per_sec_per_gpu  ({nproc} GPUs) : {metrics['tokens_per_sec_per_gpu']:.2f}")
print(f"train_samples_per_second          : {metrics['train_samples_per_second']:.2f}")
print(f"train_steps_per_second            : {metrics['train_steps_per_second']:.4f}")
print(f"train_runtime (s)                 : {metrics['train_runtime']:.2f}")
print(f"train_loss                        : {metrics['train_loss']:.4f}")

# 2) 写入 /workspace/results/result.json
os.makedirs(os.path.dirname(result_path), exist_ok=True)
result = {'status': 'success', 'metrics': metrics}
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# 3) 把 result.json 内容回显到 stdout（必须与文件路径在同一行，
#    便于上层 mcp__agent 通过 "result.json" 关键字 + {...} 正则提取）
print(f"result.json: {json.dumps(result, ensure_ascii=False)}")
EOF
```

**输出示例**（基于上方样例日志）：
```
train_tokens_per_second (total)   : 0
tokens_per_sec_per_gpu  (8 GPUs)  : 0
train_samples_per_second          : 0
train_steps_per_second            : 0
train_runtime (s)                 : 0
train_loss                        : 0
result.json: {"status": "success", "metrics": {"train_tokens_per_second": 0, "tokens_per_sec_per_gpu": 0, "train_samples_per_second": 0, "train_steps_per_second": 0, "train_runtime": 0, "train_loss": 0}}
```

**结果文件**（`/workspace/results/result.json`）：
```json
{
  "status": "success",
  "task": "nlp_finetune",
  "gpu_count": 8,
  "log": "/workspace/logs/finetune_128_4_closeint8.log",
  "metrics": {
    "train_tokens_per_second": 0,
    "tokens_per_sec_per_gpu": 0,
    "train_samples_per_second": 0,
    "train_steps_per_second": 0,
    "train_runtime": 0,
    "train_loss": 0
  }
}
```

## 常见问题

1. **找不到批量脚本**：确认 `/workspace/scripts/batch_finetune.sh` 已由 skill 资源注入；复制到代码目录或从预置路径直接执行。
2. **找不到训练入口**：确认 `$CODE_DIR/alpaca-lora/finetune_int8_close.py` 存在，或设置 `NLP_FIN_ALPACA_DIR`。`finetune.sh` 不再需要。
3. **`TypeError: ... unexpected keyword argument 'evaluation_strategy'`**：新版 transformers 把 `TrainingArguments` 的 `evaluation_strategy` 改名为 `eval_strategy`。在 `finetune_int8_close.py` 里把 `evaluation_strategy=` 改成 `eval_strategy=`。
4. **`exit code 127` / `No such file or directory: .venv/bin/python`**：`PYTHON` 默认值不可用。确认容器 PATH 里有 `python3`（镜像 `nvidia-nlp-finetune:latest` 有），或显式 `export PYTHON=/opt/conda/bin/python3`。
5. **没有生成预期日志**：确认 `LOGS_DIR` 可写；切换 batch/micro-batch 时同时设置 `NLP_FIN_LOG_FILE`。
6. **日志未更新**：脚本只接受本次运行后更新的日志。检查训练是否提前失败（`tail -50 $NLP_FIN_LOG_FILE` 看错误）。
7. **找不到汇总行**：确认 `$CODE_DIR/trainer.py` 和 `$CODE_DIR/trainer_utils.py` 存在——`batch_finetune.sh` 会把它们覆盖到容器内 `transformers` site-packages，Trainer 汇总行才会有 `train_tokens_per_second`。覆盖失败时看日志里的 `Patched transformers/...` / `WARNING: ... not found` 行。另外训练在 save/load_best_model 阶段崩的话也不会有汇总行——看日志末尾的 traceback。
8. **单卡吞吐错误**：让 `NLP_FIN_NGPU` 与 `torch.distributed.run --nproc_per_node` 保持一致。
9. **模型路径错误**：`MODEL_DIR` 默认 `/data/models/llama-7b-hf`，要求该目录下直接有 `config.json`（扁平布局）或 `snapshots/<commit_hash>/config.json`（HF 缓存布局）。若是缓存布局，可显式 `export MODEL_DIR=/data/models/llama-7b-hf/snapshots/<commit_hash>`。
