---
name: template-skill
description: >
  替换为 skill 的描述，说明该评测 skill 的作用以及触发条件。
  应为一到两句完整的话，清楚说明评测目标（芯片平台、模型、任务类型）和适用场景。
user-invokable: true # 可选
compatibility: "NVIDIA GPU / Hygon DCU / 其他芯片" # 可选
metadata: # 可选
  version: "1.0.0" # 可选
  category: training  # 可选: training, finetune, inference, operator
  scenario: nlp       # 可选: nlp, cv, mm, science, audio
  tags: [benchmark, training, nvidia] # 可选
---

# Skill 名称

## 概述

简要说明：
- 该评测 Skill 解决什么问题（评测什么芯片上的什么任务）
- 何时触发（用户的关键触发词或场景）
- 预期的输入（模型、数据集、配置）和输出（性能指标、日志、报告）

## 硬件要求

- 芯片类型与数量要求
- 显存/内存要求
- 存储要求

## 依赖要求

- Docker 镜像
- 预装的关键依赖（框架、算子库等）

## 环境变量

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `MODEL_DIR` | `/data/models` | 是 | 模型权重目录 |
| `DATASET_DIR` | `/data/datasets` | 是 | 数据集目录 |
| `OUTPUT_DIR` | `/workspace/results` | 是 | 评测结果输出目录 |

## 执行流程

### 步骤 1：容器启动

```bash
docker run -it \
  --name benchmark_container \
  --gpus all \
  --shm-size=128g \
  -v $MODEL_DIR:/data/models:ro \
  -v $DATASET_DIR:/data/datasets:ro \
  -v $OUTPUT_DIR:/workspace/results:rw \
  <image_name> \
  /bin/bash
```

### 步骤 2：数据准备

按需描述数据预处理步骤。

### 步骤 3：执行评测

```bash
cd /workspace/code
bash run_benchmark.sh 2>&1 | tee /workspace/logs/benchmark.log
```

## 关键性能指标

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `tokens_per_sec_per_gpu` | 每 GPU 每秒处理 token 数 |
| 性能（辅助） | `GPU 利用率` | GPU 使用率 |
| 资源（辅助） | `内存使用` | 显存占用 |
| 精度（选采） | `RMSE / ACC` | 结果精度指标 |

## 示例

**示例 1：典型用法**
```
用户说： "帮我评测 nvidia 上的 nlp 训练性能"
执行：
  1. 检查模型与数据集路径
  2. 启动容器
  3. 执行评测脚本
  4. 采集性能指标
结果：输出 tokens_per_sec_per_gpu 等核心指标
```

**示例 2：自定义配置**
```
用户说： "用 Qwen3-8B 在 4 卡上测试训练吞吐"
执行：
  1. 根据用户指定修改 GPU 数量和模型配置
  2. 调整脚本参数
  3. 执行评测并输出结果
结果：输出对应配置下的性能指标
```

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 容器启动失败 | GPU 设备不可用 | 检查 GPU 驱动和 nvidia-docker |
| 显存不足 | 模型或 batch_size 过大 | 减小 batch_size 或更换 GPU |
| 数据集找不到 | 路径配置错误 | 检查环境变量和挂载路径 |
| 训练/推理超时 | 集群资源不足 | 检查节点可用性和网络状态 |
