# nvidia-nlp-inference 多机启动模板提示

本文件由 SKILL.md frontmatter `multi_host_hint` 引用，在 `nnodes > 1` 时会被
拼入 Generator 的 system prompt。这里只补充 **sglang 跨机推理特有** 的内容，
与 Generator 通用多机段落（result.json 路径、ready check、blocking 约定等）
配合使用。

## 典型规模

- DeepSeek-R1 671B 16 卡推理：`nnodes=2`、每台 8×H200、`WORLD_SIZE=16`
- 总并行度 `--tp = WORLD_SIZE`（即 16），跨两机做张量并行
- 单机情形（`nnodes=1`）请直接走 SKILL.md 主流程，不要套用本模板

## 步骤拆分（CommandGroup）

建议生成 3 个 step（不要合并到一个脚本里）：

| step_id | target | blocking | 作用 |
|---|---|---|---|
| `launch_server` | `all` | `false` | 每台机器内部 `nohup` 起 `sglang.launch_server`，前台 `until` 探活，仅 rank0 暴露 HTTP；ready 后 `exit 0` |
| `bench` | `rank0` | `true` | 服务就绪后跑 `sglang.bench_serving` 压测，写 `bench.log` / `bench.csv` |
| `collect_metrics` | `rank0` | `true` | 解析 `bench.log`，写 `/workspace/results/result.json` |

`metric_source` 指向 `collect_metrics`。

### launch_server step（rank-aware 模板）


```bash
#!/bin/bash
# --- 残留清理（防止上次 nohup 占住 GPU）---
pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
pkill -9 -f 'sglang.srt'           2>/dev/null || true
sleep 3

# --- NVSHMEM（sglang 跨机 MoE / EP 通信走 NVSHMEM IBGDA）---
export NVSHMEM_HCA_LIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IBGDA_NUM_RC_PER_PE=8
export NVSHMEM_IB_TRAFFIC_CLASS=186
export NVSHMEM_DISABLE_NVLs=1

# --- NCCL（sglang 跨机 TP / allreduce 走 NCCL over IB）---
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=186
export NCCL_NVLS_ENABLE=0

mkdir -p /workspace/logs

# /data/models/snapshots/<commit_hash> 走容器挂载点，由 docker run -v 决定）
MODEL_PATH=/data/models/snapshots/4236a6af538feda4548eca9ab308586007567f52

nohup python3 -m sglang.launch_server \
  --model ${MODEL_PATH} \
  --dist-init-addr ${MASTER_ADDR}:${MASTER_PORT} \
  --nnodes ${NNODES} \
  --node-rank ${NODE_RANK} \
  --tp ${WORLD_SIZE} \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  > /workspace/logs/serve.rank${NODE_RANK}.log 2>&1 &
SERVER_PID=$!
echo ${SERVER_PID} > /workspace/logs/serve.rank${NODE_RANK}.pid

# --- ready check：进程存活 + 业务就绪信号双重确认，禁止仅靠 sleep N ---
TIMEOUT=2400   # 40 分钟，足够 671B 跨机加载 + capture cuda graph
ELAPSED=0
while [ ${ELAPSED} -lt ${TIMEOUT} ]; do
  if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "ERROR: rank${NODE_RANK} server pid ${SERVER_PID} died" >&2
    tail -n 200 /workspace/logs/serve.rank${NODE_RANK}.log >&2
    exit 1
  fi
  if [ "${NODE_RANK}" = "0" ]; then
    # rank0 暴露 HTTP，用 /v1/models 判定真就绪
    if curl -fs -m 5 http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
      echo "rank0 server ready after ${ELAPSED}s"
      exit 0
    fi
  else
    # 非 rank0 不暴露 HTTP，用日志关键字 + 进程存活双重确认
    if grep -q "Capture cuda graph end\|The server is fired up" \
         /workspace/logs/serve.rank${NODE_RANK}.log 2>/dev/null; then
      echo "rank${NODE_RANK} worker ready after ${ELAPSED}s"
      exit 0
    fi
  fi
  sleep 10
  ELAPSED=$((ELAPSED + 10))
done
echo "ERROR: rank${NODE_RANK} not ready after ${TIMEOUT}s" >&2
tail -n 200 /workspace/logs/serve.rank${NODE_RANK}.log >&2
exit 1
```

关键点：

- **`--tp = ${WORLD_SIZE}`**（不是 `${GPUS_PER_NODE}`）；跨机 TP，全卡协同。
  例：2×8 H200 → `WORLD_SIZE=16` → `--tp 16`
- **`--dist-init-addr ${MASTER_ADDR}:${MASTER_PORT}`** 由 Executor 注入，
  不要自己另起端口或写死 IP（人工脚本里写死的 `10.102.97.33:20000` 在 agent
  评测下由 Executor 用 rank0 的 ssh_host + 实时扫描出的空闲端口替换）
- **`--nnodes ${NNODES}` / `--node-rank ${NODE_RANK}`** 必填，sglang 据此识别
  当前进程在分布式拓扑中的位置
- **NCCL / NVSHMEM env 一定要 export 在 `python3 -m sglang.launch_server`
  之前**；这两组参数控制 RoCE/IB 走哪些 HCA、用哪个 GID、走哪条流量类。
  缺失或值不对会直接 NCCL/NVSHMEM init 失败或 hang
- **`NCCL_IB_HCA="=mlx5_0,..."`** 等号前缀是 NCCL 的精确匹配语法（不带等号
  会按前缀匹配，可能命中 `mlx5_bond_*` 这类非物理网卡），**不能省**
- **不要加 `-p host:container` 端口映射**：容器已经是 `--network=host`，
  端口 30000 直通宿主网络
- **per-rank 文件必须带 `${NODE_RANK}` 后缀**：`serve.log` / `serve.pid` 等
  通常挂在共享 NFS / GPFS / Lustre 下，不加后缀会两台机器互相覆盖
- **`--model`** 与 `--model-path` 在当前 sglang 版本是同义；本模板沿用人工
  评测脚本写法（`--model`）

### bench step（仅 rank0）

```bash
#!/bin/bash
set -e

HOST=127.0.0.1
PORT=30000
INPUT_LEN=2048
OUTPUT_LEN=2048
NUM_PROMPTS=1000

mkdir -p /workspace/logs

python3 -m sglang.bench_serving \
  --model /data/models/snapshots/4236a6af538feda4548eca9ab308586007567f52 \
  --random-range-ratio 1 \
  --backend sglang \
  --dataset-name random \
  --dataset-path /data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --output-file /workspace/logs/bench.csv \
  --seed 42 2>&1 | tee /workspace/logs/bench.log
```

约束：

- 必须 `target=rank0` 且 `blocking=true`
- `--host 127.0.0.1`：bench 与 rank0 server 同容器，走 host 网络 loopback
- `bench.log` / `bench.csv` 只 rank0 写，**不需要** `${NODE_RANK}` 后缀
- `depends_on: ["launch_server"]`

### collect_metrics step（仅 rank0）

正则提取 `bench.log` 末尾的性能汇总，按 SKILL.md 「步骤 4」的 Python 脚本写
`/workspace/results/result.json`。**唯一允许的路径**就是这个，写到其它路径
（如 `/workspace/result.json` 少一个 s）agent 不会收集。

关键调整（相对单机）：
- `tp = WORLD_SIZE` 而不是单机的 8；`output_tokens_per_sec_per_gpu = output_throughput / WORLD_SIZE`
- `log_path = '/workspace/logs/bench.log'`（rank0 那份，**不**带 `_rank0` 后缀）
- `depends_on: ["bench"]`，`blocking=true`，**它就是组内的终端 step**

## 跨机协调

- 严禁 `ssh` / `scp` 到其它机器，所有同步走 `${MASTER_ADDR}:${MASTER_PORT}` socket
- 每个 step 的 `target`、`depends_on` 让 Runner 帮你做拓扑同步，不要在脚本里
  自己 `while !; do sleep`、`flock` 跨机文件锁

## 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `memory unbalanced` / OOM at start | 上一次 nohup 服务还活着占着 GPU | 脚本开头先 `pkill -9 -f sglang.launch_server` |
| `NCCL ... timeout` 跨机 | IB 网络参数没设对 / HCA 名错 | 严格按上方 NCCL_* / NVSHMEM_* 一组 env 替换 |
| `NCCL_IB_HCA` 命中 bond 虚拟口而非物理 HCA | 没写 `=` 前缀，按前缀匹配错位 | 用 `NCCL_IB_HCA="=mlx5_0,..."` 精确匹配 |
| ready check 误判 → bench step 立即报 connection refused | 只检端口 listen，没等模型加载完 | ready check 改成命中 `/v1/models` / `/health` |
| rank0 写 result.json 失败 | 容器内 `/workspace/results` 不可写 | 检查 `-v $RESULTS_DIR:/workspace/results:rw` 挂载 |
