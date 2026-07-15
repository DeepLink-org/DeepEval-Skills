---
name: nvidia-nlp-finetune-mixtral
description: NVIDIA GPU 上 Mixtral-8x7B-Instruct-v0.1 微调评测技能。用于使用 xtuner、OpenAssistant Guanaco 数据集和 finetune.sh 执行单机或多机微调、采集训练日志和性能指标。
metadata:
  test_case: mixtral
  multi_host_hint: references/multi_host.md
---

# nvidia-nlp-finetune: mixtral

本 SKILL.md 描述单机 8 卡 Mixtral 微调流程。多机评测请同时遵循 `references/multi_host.md`；该文件会在 `nnodes > 1` 时自动加入 Generator prompt。

训练启动脚本是本 Skill 自带的 `scripts/finetune.sh`。Executor 会把它预置到容器内 `/workspace/scripts/finetune.sh`。

## 触发条件

当用户表达以下需求时使用：

- “在 NVIDIA 上微调 Mixtral”
- “运行 Mixtral-8x7B-Instruct-v0.1 finetune”
- “用 OpenAssistant Guanaco 训练 Mixtral”
- “执行 Mixtral Full 微调性能评测”
- “测试 Mixtral 单机 8 卡或双机 16 卡训练吞吐”

## 硬件要求

- 单机：1 节点，8 张 NVIDIA GPU
- 多机：每节点 8 张 NVIDIA GPU
- 容器使用 host 网络和 host IPC

## 依赖要求

**Docker 镜像**：

```text
registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:zs-test
```

镜像中应包含 PyTorch、分布式训练依赖和运行当前 xtuner 代码所需的 Python 环境。

## 环境变量

### 环境变量定义

| 环境变量 | 容器内目录 | 是否必需 | 说明 |
|---|---|---|---|
| `MODEL_DIR` | `/data/models` | 是 | Mixtral 模型权重目录 |
| `CODE_DIR` | `/workspace/code` | 是 | 训练配置、calc.py 和 xtuner 代码 |
| `DATASET_DIR` | `/data/datasets` | 是 | OpenAssistant Guanaco 数据集目录 |
| `RESULTS_DIR` | `/workspace/results` | 是 | agent 收集的 `result.json` |
| `LOGS_DIR` | `/workspace/logs` | 是 | launcher、训练和指标采集日志 |

宿主机路径由 `config/skills/NVIDIA/nlp/finetune.json` 中
`test_cases.mixtral.environment` 提供，不在 Skill 文档中硬编码。

### 目录结构

容器内 `/data/models` 应至少包含：

```text
/data/models/
├── config.json
├── generation_config.json
├── model-00001-of-00019.safetensors
├── ...
├── model-00019-of-00019.safetensors
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer.model
├── tokenizer_config.json
└── special_tokens_map.json
```

容器内 `/workspace/code` 应包含：

```text
/workspace/code/
├── calc.py
├── mixtral_8x7b_instruct_full_oasst1_e3_copy2.py
└── xtuner/
```

宿主机的 `DATASET_DIR` 已经指向 `openassistant-guanaco` 目录本身，因此挂载后，数据集
文件应直接位于容器内 `/data/datasets`：

```text
/data/datasets/
└── <OpenAssistant Guanaco 数据文件>
```

## 执行流程

### 步骤 1：启动容器

启动前在宿主机检查并创建挂载目录：

```bash
test -d "$MODEL_DIR"
test -d "$CODE_DIR"
test -d "$DATASET_DIR"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
```

容器启动模板：

```bash
docker run -d \
  --name finetune_mixtral \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=128g \
  -v $MODEL_DIR:/data/models/:ro \
  -v $CODE_DIR:/workspace/code/:rw \
  -v $DATASET_DIR:/data/datasets/:ro \
  -v $RESULTS_DIR:/workspace/results/:rw \
  -v $LOGS_DIR:/workspace/logs/:rw \
  registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:zs-test \
  tail -f /dev/null
```

多机评测的容器命名、拓扑环境变量注入和完整启动模板见
`references/multi_host.md`。单机流程不要加入多机变量。

#### 容器管理

```bash
docker exec -it finetune_mixtral bash

# 容器已停止时
docker start finetune_mixtral
docker exec -it finetune_mixtral bash
```

### 步骤 2：验证环境

进入容器后执行：

```bash
set -e
nvidia-smi

test -f /data/models/config.json
test -f /data/models/model.safetensors.index.json
test "$(find /data/models -maxdepth 1 -name 'model-*-of-00019.safetensors' | wc -l)" -eq 19

test -f /workspace/scripts/finetune.sh
test -f /workspace/scripts/calc.sh
test -f /workspace/code/mixtral_8x7b_instruct_full_oasst1_e3_copy2.py
test -d /workspace/code/xtuner
test -d /data/datasets
test -n "$(find /data/datasets -mindepth 1 -maxdepth 1 -print -quit)"

mkdir -p /workspace/results /workspace/logs
```

检查训练配置中的模型和数据集路径：

```bash
cd /workspace/code
grep -nE 'Mixtral|pretrained|model_path|openassistant|guanaco|data(_root|_path)?' \
  mixtral_8x7b_instruct_full_oasst1_e3_copy2.py || true
```

配置使用的模型和数据路径必须与容器挂载一致：

```text
/data/models
/data/datasets
```

如果配置仍指向宿主机绝对路径，必须在运行前改成容器路径。

### 步骤 3：执行单机评测

可在容器内任意工作目录执行；Skill 脚本会自行进入 `/workspace/code`：

```bash
set -o pipefail

NODE_COUNT=1 PROC_PER_NODE=8 \
  bash /workspace/scripts/finetune.sh 2>&1 | tee /workspace/logs/launcher.rank0.log
```

注意：

- 不要绕过 Skill 自带的 `finetune.sh` 直接运行 Python 配置；
- 默认执行 Full 配置 `mixtral_8x7b_instruct_full_oasst1_e3_copy2.py`；
- 如需切换配置，显式设置 `MIXTRAL_CONFIG`，且配置文件必须位于 `/workspace/code`；
- `finetune.sh` 非零退出时立即判定失败；
- launcher 日志固定写到 `/workspace/logs/launcher.rank0.log`；
- 详细训练日志由 Skill 脚本直接写入
  `/workspace/logs/train_Full_8_node0_<timestamp>.log`；
- `/workspace/logs/train_Full_node0.path` 记录本次实际训练日志的完整路径。

**输出产物**：

| 文件 | 容器内路径 | 说明 |
|---|---|---|
| launcher 日志 | `/workspace/logs/launcher.rank0.log` | 外层启动过程的 stdout/stderr |
| 训练日志 | `/workspace/logs/train_Full_8_node0_<timestamp>.log` | xtuner 的完整 stdout/stderr |
| 训练日志路径文件 | `/workspace/logs/train_Full_node0.path` | 本次训练日志的唯一定位信息 |
| 结构化结果 | `/workspace/results/result.json` | 指标采集步骤生成 |

训练结束后检查：

```bash
test -s /workspace/logs/launcher.rank0.log
test -s /workspace/logs/train_Full_node0.path
TRAIN_LOG="$(cat /workspace/logs/train_Full_node0.path)"
case "$TRAIN_LOG" in
  /workspace/logs/train_Full_8_node0_*.log) ;;
  *) echo "unexpected single-node training log: $TRAIN_LOG" >&2; exit 1 ;;
esac
test -s "$TRAIN_LOG"
```

### 步骤 4：指标采集

单机训练成功后执行 Skill 自带的指标脚本：

```bash
TRAIN_LOG="$(cat /workspace/logs/train_Full_node0.path)"
bash /workspace/scripts/calc.sh "$TRAIN_LOG" 5 22
```

脚本执行以下固定流程：

1. 接收本次实际训练日志 `/workspace/logs/train_Full_8_node0_<timestamp>.log`；
2. 只接受 `/workspace/logs/train_Full_*_node0_*.log`，拒绝其他路径；
3. 调用 `python /workspace/code/calc.py "$filename" 5 22`；
4. 校验 stdout 是单个有限数值，将其作为 `throughput` 写入
   `/workspace/results/result.json`。

默认 `start_iter=5`、`end_iter=22`。如果评测协议改变迭代区间，必须显式传入新值，
不能自行推断。

输出数值经有限浮点数校验后写入 `/workspace/results/result.json`：

```json
{
  "status": "success",
  "metrics": {
    "throughput": 0
  }
}
```

其中 `0` 只是格式示例，实际值必须完全来自 `calc.py` stdout。

## 失败条件

以下任一情况不得标记成功：

- 模型 index 或任一 safetensors 分片缺失；
- 数据集、xtuner、训练脚本或配置文件缺失；
- 训练配置仍指向容器内不可访问的宿主机路径；
- `finetune.sh` 返回非零；
- 日志包含 traceback、CUDA OOM、NCCL timeout 或任一 rank 失败；
- 训练日志路径文件缺失，或 `calc.sh` 收到的训练日志不存在、为空或路径不合法；
- `calc.py` 输出不是单个有限数值；
- `/workspace/results/result.json` 缺失、格式错误或 metrics 为空。
