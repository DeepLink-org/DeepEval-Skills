# Mixtral 微调多机执行提示

本文件只在 `nnodes > 1` 时加载。
Executor 会把 rank0 的地址注入`${MASTER_ADDR}`，并为每个节点注入 `${NODE_RANK}`。

## 多机容器启动

集群拓扑由 Scheduler/Executor 决定，不由 `finetune.sh` 猜测。Creator 生成
`docker run` 模板时必须把拓扑变量注入容器，并返回不带 rank 的基础容器名：

```bash
test -d "$MODEL_DIR"
test -d "$CODE_DIR"
test -d "$DATASET_DIR"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
```

Creator 输出的容器启动模板使用基础名称；Executor 会在每台机器上自动追加 `_rank<i>`：

```bash
docker run -d \
  --name finetune_mixtral \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=128g \
  -e NODE_RANK=$NODE_RANK \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e MASTER_PORT=$MASTER_PORT \
  -e WORLD_SIZE=$WORLD_SIZE \
  -e NNODES=$NNODES \
  -e GPUS_PER_NODE=$GPUS_PER_NODE \
  -v $MODEL_DIR:/data/models/:ro \
  -v $CODE_DIR:/workspace/code/:rw \
  -v $DATASET_DIR:/data/datasets/:ro \
  -v $RESULTS_DIR:/workspace/results/:rw \
  -v $LOGS_DIR:/workspace/logs/:rw \
  registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:zs-test \
  tail -f /dev/null
```

这里的 `$NODE_RANK`、`$MASTER_ADDR` 等是 Creator 输出模板中的占位符，Executor 会在
每台机器上替换为本次调度的真实值。Docker 不会自动继承宿主机变量，因此必须通过
`-e` 跨越容器边界注入。

变量职责：

| 变量 | 来源 | 作用 |
|---|---|---|
| `NODE_RANK` | Executor 按节点分配 | 当前节点的唯一 rank |
| `MASTER_ADDR` | Executor 取 rank0 地址 | 分布式 rendezvous 地址 |
| `MASTER_PORT` | Executor 为本次任务选择 | 分布式 rendezvous 端口 |
| `NNODES` | Scheduler/Executor | 总节点数 |
| `GPUS_PER_NODE` | Scheduler/Executor | 每节点进程/GPU 数 |
| `WORLD_SIZE` | Scheduler/Executor | 全局进程数，通常为 `NNODES × GPUS_PER_NODE` |

`finetune.sh` 接受 `NNODES`/`GPUS_PER_NODE`，也兼容 `NODE_COUNT`/`PROC_PER_NODE`，
并转换为训练入口使用的 `ADDR`、`PORT`、`NPROC_PER_NODE` 等变量。它不负责选择
master、rank 或端口。

Creator 返回基础容器名 `finetune_mixtral`；Executor 实际创建
`finetune_mixtral_rank0`、`finetune_mixtral_rank1` 等容器，便于独立执行、清理和定位。

## CommandGroup

生成两个 step：

| step_id | target | blocking | depends_on | 作用 |
|---|---|---:|---|---|
| `train` | `all` | `true` | `[]` | 所有节点同时执行分布式微调 |
| `collect_metrics` | `rank0` | `true` | `["train"]` | 汇总日志并生成 `result.json` |

`metric_source` 必须指向 `collect_metrics`。

### train step

所有节点在各自容器内执行同一份 rank-aware 脚本：

```bash
#!/bin/bash
set -e
set -o pipefail

mkdir -p /workspace/logs /workspace/results

pkill -9 -f 'xtuner train' 2>/dev/null || true
sleep 3

export NODE_RANK="${NODE_RANK}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export NODE_COUNT="${NNODES}"
export PROC_PER_NODE="${GPUS_PER_NODE}"

test "${NODE_COUNT}" -gt 1
test "${PROC_PER_NODE}" -gt 0

bash /workspace/scripts/finetune.sh 2>&1 | \
  tee "/workspace/logs/launcher.rank${NODE_RANK}.log"
```

对于用户给出的双机 8 卡/节点场景，变量展开应等价于：

```text
rank0: NODE_RANK=0 NODE_COUNT=2 PROC_PER_NODE=8
rank1: NODE_RANK=1 NODE_COUNT=2 PROC_PER_NODE=8
```

这要求任务的总 `card_count=16`；Executor 会据此计算
`GPUS_PER_NODE=card_count / NNODES=8`。

两台机器必须近同时启动。不要先等待 rank0 完成再启动 rank1，也不要在脚本中通过
`ssh`/`scp` 启动另一节点。

#### 容器要求

- 使用 `--network=host`；
- 每节点使用全部 8 张 GPU；
- Creator 返回的基础容器名为 `finetune_mixtral`，实际 rank 后缀由 Executor 添加；
- Creator 必须通过 `-e` 传入 `NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`、
  `WORLD_SIZE`、`NNODES` 和 `GPUS_PER_NODE`；
- 两节点使用相同模型、数据集、代码版本和训练配置；
- 如果日志目录位于共享文件系统，所有节点写入的文件必须包含 rank 后缀。

Skill 脚本读取 `MASTER_PORT`，缺省值是 `29600`。两个节点必须收到相同的
`MASTER_ADDR` 和 `MASTER_PORT`。

### collect_metrics step

该 step 只在 rank0、且所有 train step 成功之后执行：

```bash
#!/bin/bash
set -e
set -o pipefail

cd /workspace/code

test -s /workspace/logs/launcher.rank0.log
test -s /workspace/logs/launcher.rank1.log
test -s /workspace/logs/train_Full_node0.path
test -s /workspace/logs/train_Full_node1.path

TRAIN_LOG="$(cat /workspace/logs/train_Full_node0.path)"
WORKER_LOG="$(cat /workspace/logs/train_Full_node1.path)"
case "$TRAIN_LOG" in
  /workspace/logs/train_Full_16_node0_*.log) ;;
  *) echo "unexpected rank0 training log: $TRAIN_LOG" >&2; exit 1 ;;
esac
case "$WORKER_LOG" in
  /workspace/logs/train_Full_16_node1_*.log) ;;
  *) echo "unexpected rank1 training log: $WORKER_LOG" >&2; exit 1 ;;
esac
test -s "$TRAIN_LOG"
test -s "$WORKER_LOG"

if grep -Ei 'Traceback|CUDA out of memory|NCCL.*(error|timeout)|RuntimeError' \
    "$TRAIN_LOG" "$WORKER_LOG"; then
  echo "distributed training log contains fatal errors" >&2
  exit 1
fi

bash /workspace/scripts/collect_metrics.sh "$TRAIN_LOG" 5 22
```

指标采集只使用 rank0 本次训练日志，并严格执行
`python /workspace/code/calc.py "$filename" 5 22`。`calc.py` 的单个数值输出代表 16 卡训练
吞吐量；不要再与 rank1 结果相加，否则会重复计算。

最终只由 rank0 写：

```text
/workspace/results/result.json
```

并校验 `status == "success"`、`metrics` 是非空对象。任一 rank 训练失败时，跳过成功指标生成。

## 常见错误

| 现象 | 检查项 |
|---|---|
| 两节点互相等待 | `MASTER_ADDR` 是否都为 rank0 地址；`NODE_RANK` 是否分别为 0/1；端口是否一致 |
| 只有 8 个进程 | `NODE_COUNT=2`、每节点 `PROC_PER_NODE=8` 是否同时传入 |
| 日志被覆盖 | 共享日志文件是否带 `.rank${NODE_RANK}` 后缀 |
| calc 得到旧结果 | 是否使用 rank0 的 `train_Full_node0.path` 指向的本次训练日志 |
| 结果不是数值 | `calc.py` stdout 必须仅包含吞吐量数值，不能打印额外说明 |
| 结果翻倍 | `calc.py` 的 rank0 输出已经是 16 卡吞吐量，不要再与 rank1 相加 |
