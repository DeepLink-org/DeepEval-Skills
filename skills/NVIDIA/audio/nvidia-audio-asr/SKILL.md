---
name: nvidia-audio-asr
description: NVIDIA GPU 上语音识别模型推理性能评测技能。用于指导 executor 完成容器启动、批量检测训练脚本执行、日志采集与性能指标分析。
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 ASR 模型测试"
- "帮我测试 SenseVoice 多语言语音理解模型推理性能"
- "在 nvidia 上跑 中文语音识别 Aishell-1 学术数据集测试集"
- "帮我批量测试语音识别模型 FP32/FP16 性能"
- "采集 ASR 模型 CER (字符错误率)"

---

**基础目录配置**：
- 模型权重目录：`/data/models`
- 数据集目录：`/data/datasets`
- 代码挂载目录：`/workspace/code`
- 配置文件目录：`/workspace/config`
- 推理日志输出目录：`/workspace/logs`
- 测试结果输出目录：`/workspace/results`
  - 测试指标输出文件：`/workspace/results/acc_report.json`
  - 测试数据预测结果目录：`/workspace/results/predictions`

---

### 支持的模型配置

**当前支持模型**：
- **SenseVoiceSmall**

**当前支持任务**：
- **模型推理性能测试**：基于 Aishell-1 数据集测试性能

**硬件要求**：
- 1 张 NVIDIA GPU
- 足够显存支撑 ASR 模型推理

### 依赖要求

依赖通过指定 Docker 镜像提供，不需要在宿主机额外安装：

```bash
117.48.149.97:5000/eval-test/asr-eval:v2
```

---

### 模型与数据路径

当前脚本默认使用以下资源：

**模型路径**：
```bash
/data/models/speech_recognition/SenseVoiceSmall
```

**数据集路径**：
```bash
/data/datasets/speech_recognition
```

如模型版本或数据集路径发生变化，应同步修改 `scripts/test.sh` 和 `/workspace/config/data_set.cfg` 中的路径。

---

### 容器启动脚本

**Docker 运行命令**：
```bash
docker run -it \
  --name asr-eval \
  --gpus all \
  --shm-size=128g \
  -e PYTHONUNBUFFERED=1 \
  -v /workspace/config:/workspace/config \
  -v /workspace/results:/workspace/results \
  -v /workspace/models:/data/models \
  -v /workspace/datasets/speech_recognition:/data/datasets/speech_recognition \
  -v /workspace/logs:/workspace/logs \
  117.48.149.97:5000/eval-test/asr-eval:v2
```

说明：
- 使用 **交互式** `-it` 进入 `bash`，便于在同一终端内执行脚本；如需后台常驻可改为 `-d` 并配合 `docker exec`。
- **`--shm-size=128g`**：避免大吞吐推理时共享内存不足。
- 若已存在同名容器，需先执行 `docker rm -f asr-eval` 或更换 `--name`。

---

### 测试启动脚本

服务启动脚本位于 skill 目录下的 `scripts/test.sh`，部署时拷贝到 `/workspace/code/`：

```bash
cp scripts/test.sh /workspace/code/
```

执行方式：

```bash
cd /workspace/code
bash test.sh 
```

当前默认行为：
- 创建 `/workspace/logs/` 目录
- 启动 `/workspace/infer_runner.py` 进行推理
- 指定模型ID：`--model_id sensevoice-small`
- 指定模型路径：`--model_dir /data/models/speech_recognition/SenseVoiceSmall`
- 指定结果输出路径：`--output_dir /workspace/results/predictions/`
- 指定准确率报告输出路径：`--acc_report /workspace/results/acc_report.json`
- 指定数据集配置文件：`--data_set /workspace/config/data_set.cfg`
- 输出日志到：`/workspace/logs/test.log`

---

### 关键性能指标

在每个数据集测评结束，日志会打印测评结果（每条测试样本的预测结果）和测试速度（单位：样本/秒），例如：
```text
结果保存到: /workspace/results/predictions/predictions_aishell1.jsonl
速度: 12.7 样本/秒
```

测试日志末尾会输出所有数据集结果汇总，例如：

```text
评测数据集数: 5
  aishell1: cer=0.0402, 样本数=7176
  librispeech-clean: wer=0.0000, 样本数=0
  librispeech-other: wer=0.0000, 样本数=0
  fleurs-cmn: cer=0.0000, 样本数=0
  fleurs-en: cer=0.0000, 样本数=0

acc_report: /workspace/results/acc_report.json
predictions: /workspace/results/predictions/
```

其中，`acc_report`记录了每个数据集的详细性能数据，例如

```text
{
  "records": [
    {
      "dataset": "aishell1",
      "total_samples": 7176,
      "success_samples": 7176,
      "error_samples": 0,
      "total_time": 562.835529088974,
      "avg_inference_time": 0.07838046932831788,
      "metrics": {
        "cer": 0.04018870234182996,
        "total_edits": 4268,
        "total_ref_len": 106199
      },
      "success_rate": 1.0,
      "schema_version": "1.0.0",
      "model": "SenseVoiceSmall",
      "model_type": "asr",
      "chip": "gpu",
      "device": "cuda:0 (NVIDIA H200)",
      "timestamp": "20260507_065251"
    }
  ],
  "last_updated": "20260507_065251",
  "notes": ""
}
```

关注以下指标：

| 类型 | 指标 | 说明 |
|---|---|---|
| 性能（必采） | `avg_inference_time` | 平均每条样本推理时间，核心吞吐指标 |
| 准确率（必采） | `cer` | Character Error Rate，字符错误率 |
| 准确率（必采） | `wer` | Word Error Rate，词错误率 |
| 准确率（辅助） | `success_rate` | 推理成功率 |

**注意**：若中文数据集，则只有CER指标；若英文数据集，则只有WER指标

**采集命令**：

```bash
# 访问结果报告json文件即可采集所有指标
cat /workspace/results/acc_report.json
```

---

### 常见问题

1. **容器名已存在**
   - 执行 `docker rm -f asr-eval` 后重试，或改用新容器名。

2. **测试无法启动**
   - 检查模型路径 `/data/models/speech_recognition` 下对应权重是否存在。
   - 检查 GPU 数量是否大于0。

3. **找不到模型或数据集**
   - 检查 `/data/models/` 和 `/data/datasets/` 挂载是否正确。

4. **共享内存不足**
   - 已使用 `--shm-size=128g`；若仍报错，可适当增大。

5. **日志或结果文件未生成**
   - 检查 `/workspace/logs` 写权限。
   - 检查 `tee` 重定向是否生效。