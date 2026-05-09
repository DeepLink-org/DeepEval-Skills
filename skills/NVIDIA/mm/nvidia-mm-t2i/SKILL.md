---
name: nvidia-mm-t2i
description: NVIDIA GPU 上 Stable Diffusion 文生图推理任务的评测技能。基于 TensorRT 进行推理加速，用于指导 executor 完成容器启动、模型构建、推理执行、日志采集与性能指标分析。
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 Stable Diffusion 文生图推理"
- "Stable Diffusion TensorRT 推理评测"
- "nvidia 文生图推理评测"
- "采集 Stable Diffusion 推理性能"
- "跑 sd t2i 推理"

---

**基础目录配置**：
- 模型权重目录：`/data/models`
- 代码挂载目录：`/workspace/code`
- 推理日志输出目录：`/workspace/logs`
- 临时缓存目录：`/workspace/tmp`

---

### 支持的模型配置

**当前支持模型**：
- **v1-5**: stable-diffusion-v1-5
- **v2-1**: stable-diffusion-2-1

**推理参数**：
- batch-size: 1, 2（默认 1）
- height/width: 512, 768, 960（必须是 8 的倍数）
- FP16 精度推理
- denoising-steps: 50（默认）

**硬件要求**：
- 1 张 NVIDIA GPU

---

### 依赖要求

依赖通过指定 Docker 镜像提供，不需要在宿主机额外安装：

```bash
registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:sd-infer
```

容器内已预装 onnx、transformers、diffusers、cuda-python 等。

---

### 模型与数据路径

当前脚本默认使用以下资源：

**模型路径**：
```bash
/data/models/stable-diffusion-v1-5/   # v1-5 权重
/data/models/stable-diffusion-2-1/   # v2-1 权重
```

通过 `--framework-model-dir` 参数指定。如模型路径发生变化，同步修改该参数值。

---

### 容器启动脚本

**Docker 运行命令**：
```bash
docker run -it \
  --name sd_inference \
  --gpus all \
  --shm-size=16g \
  -v /data/models:/data/models \
  -v /workspace/code:/workspace/code \
  -v /workspace/logs:/workspace/logs \
  -v /workspace/tmp:/workspace/tmp \
  registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:sd-infer \
  bash
```

说明：
- 使用 **交互式** `-it` 进入 `bash`，便于在同一终端内执行推理命令；如需后台常驻可改为 `-d` 并配合 `docker exec`。
- **`--shm-size=16g`**：避免大分辨率推理时共享内存不足。
- 若已存在同名容器，需先执行 `docker rm -f sd_inference` 或更换 `--name`。

---

### 推理脚本

推理脚本位置（由挂载的代码仓库提供）：

```bash
/workspace/code/TensorRT/demo/Diffusion/demo_txt2img.py
```

执行方式：

```bash
cd /workspace/code/TensorRT/demo/Diffusion

# v1-5 模型
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --framework-model-dir /data/models/stable-diffusion-v1-5 \
  --version 1.5 \
  --batch-size 1 \
  --height 512 \
  --width 512 \
  2>&1 | tee /workspace/logs/sd1.5_1_512.log

# v2-1 模型
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --framework-model-dir /data/models/stable-diffusion-2-1 \
  --version 2.1 \
  --batch-size 1 \
  --height 512 \
  --width 512 \
  2>&1 | tee /workspace/logs/sd_1_512.log
```

说明：
- `--framework-model-dir` 指向已挂载的模型权重目录。
- 推理日志通过 `tee` 写入 `/workspace/logs/`，便于宿主机侧采集。
- **不要修改** `denoising-steps` (=50)，否则与基线指标不可比。

---

### 关键性能指标

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

关注以下指标：

| 类型 | 指标 | 说明 |
|---|---|---|
| 性能（必采） | `Throughput` | 每秒生成图片数，核心吞吐指标 |
| 性能（辅助） | `Pipeline Latency` | 端到端流水线延迟 |
| 性能（辅助） | `CLIP / UNet / VAE Latency` | 各模块延迟，排查瓶颈用 |
| 资源（辅助） | `GPU Memory Usage` | 推理阶段 GPU 显存占用 |
| 耗时（辅助） | `laoding model time` | 模型加载与 TRT engine 构建耗时 |

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

### 常见问题

1. **容器名已存在**
   - 执行 `docker rm -f sd_inference` 后重试，或改用新容器名。

2. **切换模型版本或尺寸后结果异常**
   - TensorRT 会缓存 `onnx/` 和 `engine/` 目录下的编译产物。切换模型版本（如 v1-5 → v2-1）或更改图片尺寸时，必须先删除缓存：
     ```bash
     rm -rf /workspace/code/TensorRT/demo/Diffusion/onnx/ /workspace/code/TensorRT/demo/Diffusion/engine/
     ```
   - 或使用 `--force-engine-build` 参数强制重建 TRT engine。

3. **找不到模型权重**
   - 检查 `/data/models/stable-diffusion-v1-5/` 或 `/data/models/stable-diffusion-2-1/` 映射是否正确。
   - 检查 `--framework-model-dir` 参数路径是否与挂载一致。

4. **共享内存不足**
   - 已使用 `--shm-size=16g`；若大分辨率仍报错，可适当增大。

5. **推理日志未生成**
   - 检查 `/workspace/logs` 写权限。
   - 检查 `tee` 重定向是否生效。
