---
name: nvidia-mm-t2i
description: NVIDIA GPU 上 Stable Diffusion 文生图推理任务的评测技能。基于 TensorRT 进行推理加速，用于指导 executor 完成容器启动、模型构建、推理执行、日志采集与性能指标分析。
---

### 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 Stable Diffusion 文生图推理"
- "Stable Diffusion TensorRT 推理评测"
- "nvidia 文生图推理评测"
- "采集 Stable Diffusion 推理性能"
- "跑 sd t2i 推理"

---


### 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `SD_MODELS_DIR` | `/data/models` | 是 | 模型权重目录，存放 Stable Diffusion 模型权重文件 |
| `SD_CODE_DIR` | `/workspace/code` | 是 | 代码挂载目录，包含 TensorRT 推理脚本 |
| `SD_LOGS_DIR` | `/workspace/logs` | 否 | 推理日志输出目录 |
| `SD_TMP_DIR` | `/workspace/tmp` | 否 | 临时缓存目录 |

**说明**：
- **SD_MODELS_DIR** 存放模型权重文件，包含 stable-diffusion-v1-5 和 stable-diffusion-2-1 子目录
- **SD_CODE_DIR** 存放推理代码仓库，包含 TensorRT demo 脚本
- **SD_LOGS_DIR** 存放推理日志文件，用于后续性能指标采集
- **SD_TMP_DIR** 存放临时文件和缓存
- 表格中的"映射目录"列指明了容器启动时 `-v` 参数的挂载路径，即宿主机路径映射到容器内的路径

**目录结构说明**：

- `$SD_MODELS_DIR`: 模型权重根目录，默认结构如下：
  ```
  $SD_MODELS_DIR/
  ├── stable-diffusion-v1-5/    # v1-5 模型权重
  └── stable-diffusion-2-1/     # v2-1 模型权重
  ```

- `$SD_CODE_DIR`: 代码目录，典型结构如下：
  ```
  $SD_CODE_DIR/
  └── TensorRT/
      └── demo/
          └── Diffusion/
              ├── demo_txt2img.py      # 文生图推理脚本
              ├── onnx/                 # ONNX 模型缓存目录
              └── engine/              # TensorRT engine 缓存目录
  ```

**注意**：
- `SD_MODELS_DIR` 和 `SD_CODE_DIR` 为必需参数，必须提供
- TensorRT 会缓存 `onnx/` 和 `engine/` 目录下的编译产物，切换模型版本或图片尺寸时需要删除缓存


---


### 支持的模型配置

**当前支持模型**（共 2 个）：
- `v1-5`: stable-diffusion-v1-5，经典 Stable Diffusion 1.5 版本
- `v2-1`: stable-diffusion-2-1，Stable Diffusion 2.1 版本

**当前支持任务**：
- 基于 TensorRT 的文生图推理
- 性能评估：Throughput（吞吐）、Pipeline Latency（端到端延迟）
- 模块级延迟分析：CLIP / UNet / VAE-Dec

**推理参数**：
- batch-size: 1, 2（默认 1）
- height/width: 512, 768, 960（必须是 8 的倍数）
- FP16 精度推理
- denoising-steps: 50（默认，不可修改，否则与基线指标不可比）

**硬件要求**：
- 1 张 NVIDIA GPU

---


### 依赖要求

**Docker 镜像**：
```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-mm-t2i:latest
```

容器内已预装：
- onnx
- transformers
- diffusers
- cuda-python
- TensorRT

---


## 第一阶段：容器启动

### 选择模型与参数

启动容器前，先指定目标模型和推理参数：

```bash
# 1. 选择要测试的模型
export MODEL_VERSION="1.5"  # 可选: 1.5, 2.1

# 2. 设置推理参数
export BATCH_SIZE="1"
export HEIGHT="512"
export WIDTH="512"
```

### 容器创建命令

**挂载权限约定**：
- `:ro` — 只读，防止误修改
- `:rw` — 读写，用于输出目录（logs、tmp）和代码目录

**公共参数**（所有场景共享）：

| 参数 | 说明 |
|------|------|
| `--gpus all` | 挂载所有 NVIDIA GPU 设备 |
| `--shm-size=16g` | 共享内存大小，避免大分辨率推理时内存不足 |

**公共卷挂载**（所有场景必需）：
```bash
-v $SD_MODELS_DIR:/data/models:rw \
-v $SD_CODE_DIR:/workspace/code:rw \
-v $SD_LOGS_DIR:/workspace/logs:rw \
-v $SD_TMP_DIR:/workspace/tmp:rw
```

---

**容器创建命令**：

```bash
docker run -it \
  --name sd_inference \
  --gpus all \
  --shm-size=16g \
  -v $SD_MODELS_DIR:/data/models:rw \
  -v $SD_CODE_DIR:/workspace/code:rw \
  -v $SD_LOGS_DIR:/workspace/logs:rw \
  -v $SD_TMP_DIR:/workspace/tmp:rw \
  swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-mm-t2i:latest \
  /bin/bash
```

**注意**：
- 使用交互式 `-it` 进入 bash，便于在同一终端内执行推理命令；如需后台常驻可改为 `-d` 并配合 `docker exec`
- 若已存在同名容器，先执行 `docker rm -f sd_inference`
- `${MODEL_VERSION}` 在宿主机已通过 `export MODEL_VERSION=...` 设置

### 容器管理命令

**进入已创建的容器**：
```
# 如果容器已在运行
docker exec -it sd_inference /bin/bash

# 如果容器已停止，先启动再进入
docker start sd_inference
docker exec -it sd_inference /bin/bash
```

**验证容器环境**：
```
# 检查 GPU 设备
nvidia-smi

# 检查挂载的目录
ls -lh /data/models/
ls -lh /data/models/stable-diffusion-v1-5/
ls -lh /data/models/stable-diffusion-2-1/
ls -lh /workspace/code/
ls -lh /workspace/logs/
ls -lh /workspace/tmp/
```
---



## 第二阶段：容器中执行评测

### 步骤 1：进入推理目录

容器内所有路径已通过卷挂载固定（详见[环境变量定义](#环境变量定义)），无需额外设置环境变量。

```bash
cd /workspace/code/TensorRT/demo/Diffusion
```

### 步骤 2：执行推理

运行推理脚本，日志将保存至 `/workspace/logs/`：

```bash
cd /workspace/code/TensorRT/demo/Diffusion

# v1-5 模型
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --framework-model-dir /data/models/stable-diffusion-v1-5 \
  --version ${MODEL_VERSION:-1.5} \
  --batch-size ${BATCH_SIZE:-1} \
  --height ${HEIGHT:-512} \
  --width ${WIDTH:-512} \
  2>&1 | tee /workspace/logs/sd1.5_${BATCH_SIZE:-1}_${HEIGHT:-512}.log

# v2-1 模型
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --framework-model-dir /data/models/stable-diffusion-2-1 \
  --version {MODEL_VERSION:-2.1} \
  --batch-size ${BATCH_SIZE:-1} \
  --height ${HEIGHT:-512} \
  --width ${WIDTH:-512} \
  2>&1 | tee /workspace/logs/sd2.1_${BATCH_SIZE:-1}_${HEIGHT:-512}.log
```

**推理输出**：
- 生成的图片保存在当前工作目录下
- 日志文件保存在 `/workspace/logs/` 目录

**注意**：
- `--framework-model-dir` 指向已挂载的模型权重目录
- 推理日志通过 `tee` 写入 `/workspace/logs/`，便于宿主机侧采集
- **不要修改** `denoising-steps` (=50)，否则与基线指标不可比

**验证推理结果**：
```
# 检查生成的图片
ls -lh /workspace/code/TensorRT/demo/Diffusion/*.png

# 查看推理日志
tail -50 /workspace/logs/sd1.5_1_512.log
```

### 关键性能指标

#### 执行评估

推理日志中包含各模块延迟与吞吐汇总行，例如：

```text
|-----------------|--------------|
|     Module      |   Latency    |
|-----------------|--------------|
|      CLIP       |      1.59 ms |
|    UNet x 50    |    401.00 ms |
|     VAE-Dec     |      6.71 ms |
|-----------------|--------------|
|    Pipeline     |    409.37 ms |
|-----------------|--------------|
Throughput: 2.44 image/s

--------------------------------------------------
laoding model time is: 359.06847167015076 s
--------------------------------------------------

GPU Memory Usage: 5.01 GB
```

#### 指标说明

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | Throughput | 每秒生成图片数，核心吞吐指标 |
| 性能（辅助） | Pipeline Latency | 端到端流水线延迟 |
| 性能（辅助） | CLIP Latency | CLIP 文本编码器延迟 |
| 性能（辅助） | UNet Latency | UNet 去噪模块延迟（×50 步） |
| 性能（辅助） | VAE Latency | VAE 解码器延迟 |
| 资源（辅助） | GPU Memory Usage | 推理阶段 GPU 显存占用 |
| 耗时（辅助） | loading model time | 模型加载与 TRT engine 构建耗时 |

#### 指标采集

**采集命令**（将 `LOG` 替换为实际日志路径，如 `/workspace/logs/sd1.5_1_512.log`）：

```bash
# 核心：Throughput (image/s)
grep "Throughput" "$LOG" | awk '{print $2, $3}'

# 辅助：各模块延迟
grep -E "CLIP|UNet|VAE|Pipeline" "$LOG" | grep "ms"

# 辅助：GPU 显存占用
grep "GPU Memory Usage" "$LOG" | awk '{print $4, $5}'

# 辅助：模型加载耗时
grep "laoding model time" "$LOG"
```

---

## 常见问题

1. **Docker 容器启动失败**
   - **GPU 不可用**：确认宿主机上有 NVIDIA GPU 且驱动正常，执行 `nvidia-smi` 验证
   - **镜像拉取失败**：确认镜像仓库地址可访问，检查网络连接
   - **权限问题**：确保当前用户有访问 Docker 的权限
   - **共享内存不足**：如遇到内存错误，可增加 `--shm-size` 参数值（如 `32g`）

2. **容器内找不到 GPU 设备**
   - 验证 GPU 挂载：`nvidia-smi`
   - 检查驱动加载：`nvidia-smi` 是否能正常显示 GPU 信息
   - 确认 Docker 运行时：确保 Docker 配置了 nvidia 作为默认 runtime

3. **找不到模型权重**
   - 检查 `/data/models/stable-diffusion-v1-5/` 或 `/data/models/stable-diffusion-2-1/` 映射是否正确
   - 检查 `--framework-model-dir` 参数路径是否与挂载一致
   - 确认 `SD_MODELS_DIR` 环境变量已正确设置

4. **推理日志未生成**
   - 检查 `/workspace/logs` 写权限
   - 检查 `tee` 重定向是否生效
   - 确认 `SD_LOGS_DIR` 环境变量已正确设置

5. **GPU 显存不足**
   - **现象**：推理过程中报错 `OutOfMemoryError`、`CUDA out of memory` 或程序卡死
   - **原因**：当前 GPU 显存已被其他进程占用，或剩余显存不足以加载模型/数据
   - **解决方案**：
     step 1. **查看 GPU 使用情况**：
        ```bash
        nvidia-smi
        ```
     step 2. **指定可用 GPU**：
        通过 `CUDA_VISIBLE_DEVICES` 环境变量指定可用的 GPU ID：
        ```bash
        export CUDA_VISIBLE_DEVICES=0
        ```
     step 3. **重新执行脚本**
   - **注意**：
     - `CUDA_VISIBLE_DEVICES` 必须在启动 Python 脚本之前设置
     - 确保指定的 GPU ID 在容器中可见（可通过 `nvidia-smi` 确认）
