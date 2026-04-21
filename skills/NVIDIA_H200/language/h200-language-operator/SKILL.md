---
name: h200-language-operator
description: H200芯片-语言场景-算子任务的评测流程。用于指导executor完成docker容器启动、脚本生成、上传和执行的完整评测链路。参数：$0=卡数，$1=测试用例名称（如gemm、Conv2d等）。
---

# H200 语言场景 算子评测

芯片: H200 | 场景: 语言 | 任务类型: 算子
卡数: $0 | 测试用例: $1

## 第一步: 加载评测配置

读取 `references/test_configs.json`，根据参数定位配置：

1. 查找 key `"$1"`（test_case），将 card_count=$0 作为运行时参数传入
2. 若不存在，终止评测并报告：未找到 test_case=$1 的配置

从匹配的配置中获取以下字段：
- `image_name`: Docker 镜像名
- `docker_options`: docker run 附加参数（如 --gpus、--shm-size 等）
- `volumes`: 挂载目录列表（host:container 格式）
- `work_dir`: 容器内工作目录
- `task_command`: 评测执行命令
- `hints`: 补充说明（可选，仅当有无法结构化的额外信息时填写）

## 第二步: 生成 docker run 命令

根据配置字段直接拼接完整的 docker run 命令：

```
docker run -d \
  <docker_options> \
  --name h200-lang-op-$1 \
  <-v volume1> <-v volume2> ... \
  -w <work_dir> \
  <image_name> \
  sleep infinity
```

具体规则：
1. 固定前缀 `docker run -d`
2. 拼接 `docker_options`
3. 追加 `--name h200-lang-op-$1` 作为容器名
4. 遍历 `volumes` 数组，每项加 `-v` 前缀
5. 追加 `-w <work_dir>`
6. 追加 `image_name`
7. 追加 `sleep infinity` 保持容器运行

## 第三步: 启动容器

执行第二步生成的 docker run 命令。

- 执行后通过 `docker ps` 确认容器处于 running 状态
- **若失败**: 将失败命令和错误输出交给 evaluator 诊断，根据返回的 adjusted_command 和 is_recoverable 决定是否重试，最多重试 3 次

## 第四步: 生成评测脚本

在容器**外部**（宿主机），读取 `scripts/run_test.sh.tpl` 模板，进行变量替换生成最终脚本 `run_test.sh`：

替换规则：
- `{{CARD_COUNT}}` → 实际卡数 $0
- `{{TEST_CASE}}` → 实际测试用例 $1
- `{{TASK_COMMAND}}` → 配置中的 task_command
- `{{WORK_DIR}}` → 配置中的 work_dir

生成的脚本必须包含 `set -e` 以确保失败即退出。

## 第五步: 上传脚本到容器

```
docker cp run_test.sh <container>:<work_dir>/run_test.sh
docker exec <container> chmod +x <work_dir>/run_test.sh
```

## 第六步: 在容器内执行评测

```
docker exec <container> bash <work_dir>/run_test.sh
```

- 捕获标准输出和标准错误作为评测结果
- **若失败**: 将失败命令、错误输出、上下文信息交给 evaluator 诊断，根据返回结果决定是否重试或调整脚本，最多重试 3 次

## 失败处理协议

当第三步或第六步执行失败时:

1. 收集: 失败命令、完整错误输出、当前步骤上下文
2. 发送给 evaluator
3. evaluator 返回:
   - `analysis`: 原因分析
   - `adjusted_command`: 建议调整后的命令
   - `is_recoverable`: 是否可恢复
   - `suggestion`: 简短建议
4. 若 `is_recoverable=true`，使用 adjusted_command 重试，最多 3 次
5. 若 `is_recoverable=false` 或重试耗尽，终止评测并报告失败原因
