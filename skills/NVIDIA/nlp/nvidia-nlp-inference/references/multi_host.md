# nvidia-nlp-inference 多机执行流程

本文件由 SKILL.md 的 `multi_host_hint` 在 `nnodes > 1` 时注入。多机服务必须使用
`scripts/serve_multi_host.sh`；它基于已跑通的 DeepSeek-R1 跨机模板实现了残留进程清理、
跨机 TP、per-rank PID/日志和 rank-aware 就绪检查。不得在任务脚本中重新拼接
`sglang.launch_server`。

模型差异（模型路径、trust remote code、超时、压测长度、NCCL/NVSHMEM）统一从
`model_profiles.md` 的所选 profile 注入。调用前必须执行该 profile 的“多机流程”段；
脚本不会猜测网卡或 GID，也不会替模型决定单机或多机。

## 必需拓扑与 profile 参数

Executor 注入：`MASTER_ADDR`、`MASTER_PORT`、`NNODES`、`NODE_RANK`、
`GPUS_PER_NODE`。脚本计算 `WORLD_SIZE=NNODES*GPUS_PER_NODE`，并强制使用
`--tp "$WORLD_SIZE"`，而不是单机卡数。

Profile 注入：

| 参数 | 说明 |
|---|---|
| `MODEL_PATH` | 可选；未提供时自动在 `/data/models` 找到模型 |
| `TRUST_REMOTE_CODE` | `1` 时追加 `--trust-remote-code` |
| `READY_TIMEOUT` | 服务加载与 graph capture 等待秒数 |
| `INPUT_LEN`、`OUTPUT_LEN`、`NUM_PROMPTS` | benchmark 参数 |
| `DATASET_PATH`、`DATASET_PREFER` | 可选的本地 JSON 选择规则 |
| `NCCL_*`、`NVSHMEM_*` | 实际集群网络配置，必须在 launch 前 export |
| `EXTRA_SERVER_ARGS` | profile 审核后的其他 SGLang 参数 |

## CommandGroup

固定为三个步骤，`metric_source` 指向 `collect_metrics`：

| step_id | target | blocking | depends_on | 命令 |
|---|---|---:|---|---|
| `launch_server` | `all` | `false` | — | `bash /workspace/scripts/serve_multi_host.sh` |
| `bench` | `rank0` | `true` | `launch_server` | `HOST=127.0.0.1 bash /workspace/scripts/bench.sh` |
| `collect_metrics` | `rank0` | `true` | `bench` | `bash /workspace/scripts/calc.sh /workspace/logs/bench.log "$WORLD_SIZE" /workspace/logs/bench.csv` |

`launch_server` 在每台机器写 `/workspace/logs/serve.rank${NODE_RANK}.log` 和
`serve.rank${NODE_RANK}.pid`，以适配共享日志盘。仅 rank0 写不带 rank 后缀的
`bench.log`、`bench.csv`（单行 JSON summary）和唯一结果 `/workspace/results/result.json`。

## 就绪与网络约束

- rank0：进程存活且 `/v1/models` 返回成功后才允许 bench。
- 非 rank0：进程存活且服务日志出现 `Capture cuda graph end` 或 `The server is fired up` 才完成。
- 不要用固定 sleep、ssh/scp 或跨机文件锁；Runner 的 `target`、`depends_on` 负责步骤协调，SGLang 用 `${MASTER_ADDR}:${MASTER_PORT}` 协调。
- 不要添加 Docker `-p`；多机容器使用 host network。
- `NCCL_IB_HCA` 若使用物理 mlx5 网卡，必须使用 `=mlx5_0,...` 精确匹配，避免误匹配 bond 虚拟网卡。

## 故障排查

| 现象 | 处理 |
|---|---|
| OOM / GPU 不均衡 | 检查每个 rank 的清理与 `serve.rank*.log`，确认没有遗留 SGLang 服务。 |
| NCCL/NVSHMEM timeout 或 hang | 核对 profile 的 HCA、GID、traffic class 与 socket 网卡；不要跨集群复用不匹配的 DeepSeek 网络参数。 |
| bench connection refused | 查看 rank0 服务日志；必须通过 `/v1/models`，不能仅等待端口监听。 |
| 没有 result.json | 确认 `collect_metrics` 在 rank0 的 bench 后执行，并且 `/workspace/results` 可写。 |
