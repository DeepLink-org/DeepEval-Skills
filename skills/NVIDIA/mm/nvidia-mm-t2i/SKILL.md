---
name: nvidia-mm-t2i
description: "nvidia芯片-多模态场景-Stable Diffusion文生图推理任务的评测流程。基于TensorRT进行推理加速，用于指导executor完成容器启动、模型构建、推理执行和性能采集的完整评测链路。"
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 Stable Diffusion 文生图推理"
- "Stable Diffusion TensorRT 推理"
- "nvidia 文生图推理评测"

---

### 支持的模型配置

**模型版本选择**：
- **v1-5**: stable-diffusion-v1-5，权重路径：`/data/models/stable-diffusion-v1-5`
- **v2-1**: stable-diffusion-2-1，权重路径：`/data/models/stable-diffusion-2-1`

**推理参数**：
- batch-size: 1, 2（默认 1）
- height/width: 512, 768, 960
- FP16 精度推理
- denoising-steps: 50（默认）

**硬件要求**：
- 1 张 NVIDIA H200 GPU

---

### 环境准备

**代码位置**：
- `/workspace/code/TensorRT/demo/Diffusion`

**推理脚本**：
- `demo_txt2img.py` - 文生图推理入口脚本

**环境依赖**：
- onnx==1.14.0, transformers==4.31.0, diffusers==0.19.3, cuda-python 等
- 已打包在镜像 `registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:sd-infer` 中

---

### 启动配置

**Docker 运行命令**：
```bash
docker run -d --gpus all --shm-size=16g \
  -v /data/models:/data/models \
  -v /workspace/code:/workspace/code \
  -v /workspace/logs:/workspace/logs \
  -v /workspace/tmp:/workspace/tmp \
  registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:sd-infer
```

**测试命令**：
```bash
# v1-5 模型
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --framework-model-dir /data/models/stable-diffusion-v1-5 \
  --version 1.5 \
  --batch-size 1 \
  --height 512 \
  --width 512 \
  > /workspace/logs/sd1.5_1_512.log 2>&1

# v2-1 模型
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --framework-model-dir /data/models/stable-diffusion-2-1 \
  --version 2.1 \
  --batch-size 1 \
  --height 512 \
  --width 512 \
  > /workspace/logs/sd_1_512.log 2>&1
```

**参数说明**：
- `--version`: 模型版本（1.5 或 2.1）
- `--framework-model-dir`: 本地模型权重路径
- `--batch-size`: 批量大小
- `--height` / `--width`: 生成图片尺寸
- `--force-engine-build`: 强制重新构建 TRT engine（切换模型或尺寸时使用）

---

### 重要注意事项

**切换模型版本或尺寸时必须清理缓存**：

切换模型版本（如从 v1-5 切换到 v2-1）或更改图片尺寸时，必须删除之前生成的 `onnx/` 和 `engine/` 目录，否则 TensorRT 会复用缓存的 ONNX 模型和 TRT engine，导致使用错误的模型配置。

```bash
rm -rf onnx/ engine/
```

---

### 性能监控

**关键指标**：
- `Throughput` - 每秒生成图片数（image/s）

**分析命令**：
```bash
# 提取 Throughput
grep "Throughput" /workspace/logs/sd1.5_1_512.log | awk '{print $2, $3}'
```
