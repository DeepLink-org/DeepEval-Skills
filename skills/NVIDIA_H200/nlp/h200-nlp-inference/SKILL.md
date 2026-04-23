---
name: h200-nlp-inference
description: "NVIDIA H200 GPU 上基于 sglang 的 DeepSeek 文本推理评测技能。用于指导 executor 完成模型服务启动、压测脚本执行、日志采集和性能结果分析。"
---

# NVIDIA H200 NLP 推理专家

## 触发条件

当用户说以下任意内容时启动：
- "我要在 H200 上跑 DeepSeek 推理"
- "帮我测试 sglang 推理性能"
- "在 H200 上压测 DeepSeek-R1"
- "帮我启动 sglang 服务并跑 bench_serving"

---

**基础目录配置**：
- 模型权重目录：`/data/models`
- 数据集目录：`/data/datasets`
- 日志与压测结果默认输出到脚本执行目录下的日期子目录或 `logs/` 目录

---

### 支持的模型配置

**当前支持模型**：
- **DeepSeek-R1-0528**

**当前支持任务**：
- **模型服务启动**：使用 `sglang.launch_server`
- **离线压测**：使用 `sglang.bench_serving`

**硬件要求**：
- 8 张 NVIDIA H200 GPU（当前 `serve.sh` 中 `--tp 8`）
- 足够显存支撑 DeepSeek-R1 服务化推理与压测

---

### 依赖要求

需要安装并可直接调用以下 Python 模块：

```bash
python3 -m sglang.launch_server
python3 -m sglang.bench_serving
```

如果模型依赖 HuggingFace 远程代码，服务启动时需保留：

```bash
--trust-remote-code
```

压测脚本默认启用：

```bash
TRANSFORMERS_OFFLINE=1
```

---

### 模型与数据路径

当前脚本默认使用以下资源：

**模型路径**：
```bash
/data/models/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52
```

**数据集路径**：
```bash
/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

如模型版本或数据集发生变化，应同步修改 `scripts/serve.sh` 和 `scripts/test.sh` 中的路径。

---

### 服务启动脚本

服务启动脚本位置：

```bash
scripts/serve.sh
```

执行方式：

```bash
bash scripts/serve.sh
```

当前默认行为：
- 创建 `logs/` 目录
- 启动 `sglang.launch_server`
- 使用 8 卡张量并行：`--tp 8`
- 默认监听端口：`30000`
- 输出日志到：`./logs/serve.log`

---

### 压测脚本

压测脚本位置：

```bash
scripts/test.sh
```

执行方式：

```bash
bash scripts/test.sh
```

当前默认行为：
- 使用 `sglang.bench_serving` 对已启动服务进行压测
- 默认 host：`127.0.0.1`，可通过环境变量 `HOST` 覆盖
- 默认 port：`30000`，可通过环境变量 `PORT` 覆盖
- `HOST` 需要根据实际启动推理服务后暴露的 IP 地址进行调整
- 默认输入长度：`2048`
- 默认输出长度：`2048`
- 默认请求数：`2000`
- 默认结果输出到当天日期目录下

---

### 关键性能指标

关注以下指标：
- 服务是否成功启动
- 压测 CSV 是否成功生成
- 压测日志是否存在异常报错
- 吞吐、时延等 bench_serving 输出指标
- 大上下文输入下的稳定性

---

### 常见问题

1. **服务无法启动**
   - 检查模型路径是否存在
   - 检查 `sglang` 是否已正确安装
   - 检查 GPU 数量是否满足 `--tp 8`

2. **压测连接失败**
   - 检查 `serve.sh` 是否已成功启动服务
   - 检查 `test.sh` 中 `HOST` 和 `PORT` 是否与服务实际监听地址一致
   - 如果服务运行在其他节点或容器网络地址上，需要根据实际启动推理后暴露的 IP 地址调整 `HOST`

3. **找不到模型或数据集**
   - 检查 `/data/models/...` 和 `/data/datasets/...` 映射是否正确

4. **日志或结果文件未生成**
   - 检查当前目录写权限
   - 检查日期目录和 `logs/` 目录是否创建成功

5. **性能异常波动**
   - 检查请求长度、并发设置和服务负载是否稳定
