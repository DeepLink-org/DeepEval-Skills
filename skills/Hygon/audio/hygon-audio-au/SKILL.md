---
name: hygon-audio-au
description: Hygon DCU 上音频理解模型推理性能评测技能。支持基于 ECAPA-TDNN 的 lang-id-voxlingua107-ecapa 语种识别模型，用于指导 executor 完成容器启动、数据集准备、推理执行、日志采集与性能指标分析（Accuracy）。
---

### 触发条件

当用户说以下任意内容时启动：
- "我要在 Hygon 上跑音频理解模型测试"
- "帮我测试 lang-id-voxlingua107-ecapa 语种识别模型推理性能"
- "在 Hygon DCU 上跑 foundation-lid 语种识别数据集测试集"
- "帮我批量测试音频理解模型推理性能"
- "采集音频理解模型 Accuracy（语种识别准确率）"

---


### 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `AU_PROJECT_ROOT` | `/workspace` | 否 | 项目根目录，无需外部提供，模型推理代码所在目录 |
| `AU_MODEL_CKPT_DIR` | `/workspace/models` | 是 | 模型检查点目录，存放 audio_understanding 场景下的模型权重文件 |
| `AU_DATA_DIR` | `/workspace/datasets` | 是 | 数据集目录，包含 audio_understanding 场景下的数据集（如 foundation-lid 等），存放 lmdb 格式数据 |
| `AU_CONFIG_DIR` | `/workspace/config` | 是 | 配置文件目录，包含 data_set.cfg |
| `AU_INFERENCE_OUTPUT` | `/workspace/results/${MODEL_NAME}` | 否 | 推理结果输出目录 |
| `AU_LOGS_DIR` | `/workspace/logs/${MODEL_NAME}` | 否 | 日志输出目录 |

**说明**：
- **AU_PROJECT_ROOT** 不需要外部提供，镜像中已包含所有推理代码和相关工具
- **AU_MODEL_CKPT_DIR** 需要外部提供，存放 lang-id-voxlingua107-ecapa 模型权重文件
- **AU_DATA_DIR** 需要外部提供，存放预处理后的数据集，包含 audio_understanding 场景下的数据集
- **AU_CONFIG_DIR** 需要外部提供，存放 data_set.cfg 数据集路径配置文件
- 其他环境变量可根据需要自定义提供

**目录结构说明**：

- `$AU_PROJECT_ROOT`: 项目根目录（`/workspace`），默认结构如下：
  ```
  $AU_PROJECT_ROOT/                         # = /workspace
  ├── infer_runner.py                       # 推理主程序入口
  ├── inference.py                          # 推理核心逻辑
  ├── model_loader.py                       # 模型加载器
  ├── requirements.txt                      # Python 依赖清单
  ├── entrypoint.sh                         # 容器入口脚本
  ├── README.md                             # 项目说明文档
  ├── config/                               # = AU_CONFIG_DIR
  │   ├── data_set.cfg                      # 数据集路径配置文件
  │   ├── global_env_state.json             # 全局环境状态
  │   ├── prometheus.yml                    # Prometheus 监控配置
  │   └── ...
  ├── models/
  │   └── audio_understanding/              # = AU_MODEL_CKPT_DIR
  │       └── lang-id-voxlingua107-ecapa/   # ${MODEL_NAME}
  │           ├── README.md                 # 模型说明文档
  │           ├── hyperparams.yaml          # 模型超参数配置（SpeechBrain 格式）
  │           ├── embedding_model.ckpt      # ECAPA-TDNN 嵌入模型权重
  │           ├── classifier.ckpt           # 语种分类器权重
  │           ├── normalizer.ckpt           # 输入归一化参数
  │           ├── label_encoder.txt         # 语种标签编码（107 种语言）
  │           └── example/                  # 示例音频文件
  ├── datasets/                             # = AU_DATA_DIR
  │   └── audio_understanding/              # 音频理解场景数据集
  │       ├── foundation-lid/               # VoxLingua107 语种识别评测集
  │       │   ├── data_0.lmdb               # LMDB 格式数据分片 0
  │       │   ├── data_1.lmdb               # LMDB 格式数据分片 1
  │       │   └── meta.json                 # 数据集元信息
  │       └── ...
  ├── src/                                  # 评测框架源码
  ├── eval/                                 # 评测启动脚本
  ├── tests/                                # 测试用例
  ├── examples/                             # 示例代码
  ├── tutorials/                            # PyTorch 教程文档
  ├── wheeels/                              # Python wheel 包
  └── docker-examples/                      # Docker 定制示例
  ```

**关键文件说明**：
- `embedding_model.ckpt`: ECAPA-TDNN 嵌入提取模型权重，推理的核心文件
- `classifier.ckpt`: 语种分类器权重，将嵌入向量映射到 107 种语言类别
- `normalizer.ckpt`: 输入特征归一化参数
- `hyperparams.yaml`: 模型结构与推理参数配置（SpeechBrain 格式）
- `label_encoder.txt`: 语种标签到 ID 的映射，覆盖 VoxLingua107 全部 107 种语言
- `data_set.cfg`: 数据集配置文件，指定各数据集 lmdb 路径
- 每个数据集目录下包含 `data_*.lmdb`（LMDB 数据分片）和 `meta.json`（元信息）

**注意**：
- 必需的参数（如 `AU_MODEL_CKPT_DIR`、`AU_DATA_DIR`、`AU_CONFIG_DIR`）必须提供
- 表格中的"映射目录"列指明了容器启动时 `-v` 参数的挂载路径，即宿主机路径映射到容器内的路径

---

### 支持的模型配置

**当前支持模型**（共 1 个）：
- `lang-id-voxlingua107-ecapa`: 基于 ECAPA-TDNN 的语种识别模型，在 VoxLingua107 数据集上训练，支持 107 种语言的语种识别能力

**当前支持任务**：
- 基于音频理解数据集的语种识别模型推理与性能测试
- 准确率指标：Accuracy（语种识别准确率）
- 吞吐性能：`avg_inference_time`（平均推理时间/样本）

**当前支持的数据集**：
- `foundation-lid`: VoxLingua107 语种识别评测集，覆盖 107 种语言的语音片段

**硬件要求**：
- 1 张 Hygon DCU
- 足够显存支撑音频理解模型推理

---

### 依赖要求

**Docker 镜像**：
```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/hygon-audio-au:latest
```

容器内已预装：
- Python 3.10
- PyTorch（适配 Hygon DCU 版本）
- SpeechBrain（适配 Hygon DCU 版本）
- Hygon DTK 25.04.2（DCU 工具包）
- 模型推理运行器 `infer_runner.py`（位于 `/workspace/`）
- 相关 Python 依赖库

**宿主机依赖**：
- `/opt/hyhal`：Hygon HAL 库（必需，容器内只读挂载）
- `/dev/kfd`、`/dev/dri`、`/dev/mkfd`：Hygon DCU 设备文件

---

## 第一阶段：容器启动

### 容器创建命令

```bash
docker run -itd \
    --name au-eval \
    --network=host \
    --ipc=host \
    --shm-size=128g \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mkfd \
    --group-add $(stat -c '%g' /dev/dri/card0) \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e PYTHONUNBUFFERED=1 \
    -v /opt/hyhal:/opt/hyhal \
    -v $AU_CONFIG_DIR:/workspace/config:ro \
    -v $AU_MODEL_CKPT_DIR:/workspace/models:rw \
    -v $AU_DATA_DIR:/workspace/datasets:ro \
    -v $AU_INFERENCE_OUTPUT:/workspace/results/${MODEL_NAME}:rw \
    -v $AU_LOGS_DIR:/workspace/logs/${MODEL_NAME}:rw \
    swr.cn-north-1.myhuaweicloud.com/deeplink/hygon-audio-au:latest
```

**公共参数说明**：

| 参数 | 说明 |
|------|------|
| `--network=host --ipc=host` | 使用主机网络和 IPC，优化进程间通信 |
| `--shm-size=128g` | 共享内存大小，避免大吞吐推理时内存不足 |
| `--device=/dev/kfd --device=/dev/dri --device=/dev/mkfd` | 挂载 Hygon DCU 设备文件 |
| `--group-add $(stat -c '%g' /dev/dri/card0)` | 添加 DCU 设备的访问权限组 |
| `--cap-add=SYS_PTRACE` | 添加调试能力，便于性能分析 |
| `--security-opt seccomp=unconfined` | 禁用 seccomp 安全策略 |
| `-v /opt/hyhal:/opt/hyhal` | 挂载 Hygon HAL 库（必需） |
| `-e PYTHONUNBUFFERED=1` | Python 输出不缓冲，确保日志实时可见 |

**挂载权限约定**：
- `:ro` — 只读，用于输入数据（数据集、配置文件）
- `:rw` — 读写，用于输出目录（results、logs）以及 SpeechBrain 模型目录（需要写入符号链接缓存）

**注意**：
- 必须以 `-d` 参数运行（分离模式），保持容器后台运行
- 若已存在同名容器，先执行 `docker rm -f au-eval`
- `${MODEL_NAME}` 在宿主机已通过 `export MODEL_NAME=lang-id-voxlingua107-ecapa` 设置
- 模型目录挂载为 `:rw`，因为 SpeechBrain 在加载模型时可能需要在模型目录下创建符号链接或临时文件

### 容器管理命令

**进入已创建的容器**：
```bash
# 如果容器在运行
docker exec -it au-eval /bin/bash

# 如果容器已停止，先启动再进入
docker start au-eval
docker exec -it au-eval /bin/bash
```

**验证容器环境**：
```bash
# 检查 DCU 设备
ls -lh /dev/kfd /dev/dri /dev/mkfd

# 检查 HAL 库
ls -lh /opt/hyhal/lib/

# 检查 DCU 可用性
rocm-smi
# 或
hy-smi

# 检查挂载的模型权重
ls -lh /workspace/models/audio_understanding/lang-id-voxlingua107-ecapa/

# 检查数据集
ls -lh /workspace/datasets

# 检查配置文件
cat /workspace/config/data_set.cfg
```

---

## 第二阶段：容器中执行评测

### 步骤 1：进入模型目录

容器内所有路径已通过卷挂载固定（详见[环境变量定义](#环境变量定义)），无需额外设置环境变量。

```bash
# 选择要测试的模型
MODEL_NAME="lang-id-voxlingua107-ecapa"  # 可选: lang-id-voxlingua107-ecapa
DATASET_NAME="foundation-lid"            # 可选: foundation-lid
cd /workspace
```

### 配置文件说明
**配置文件关键路径**（`/workspace/config/data_set.cfg`）：
```
# lang-id-voxlingua107-ecapa 数据集配置（容器内路径）
datasets_base: "/workspace/datasets"
datasets:
  - foundation-lid
```

### 步骤 2：执行推理评测

运行推理脚本，预测结果将保存至 `/workspace/results/${MODEL_NAME}`：

```bash
cd /workspace
mkdir -p /workspace/logs/${MODEL_NAME}
python3 -u /workspace/infer_runner.py \
  --model_dir /workspace/models/audio_understanding/${MODEL_NAME} \
  --output_dir /workspace/results/${MODEL_NAME}/predictions/ \
  --acc_report /workspace/results/${MODEL_NAME}/acc_report.json \
  --data_set /workspace/config/data_set.cfg \
  --dataset ${DATASET_NAME} 2>&1 | tee /workspace/logs/${MODEL_NAME}/test.log
```

上述指令的默认行为：
- 创建 `/workspace/logs/${MODEL_NAME}` 目录
- 启动 `/workspace/infer_runner.py` 进行推理
- 指定模型路径：`--model_dir /workspace/models/audio_understanding/${MODEL_NAME}`
- 指定结果输出路径：`--output_dir /workspace/results/${MODEL_NAME}/predictions/`
- 指定准确率报告输出路径：`--acc_report /workspace/results/${MODEL_NAME}/acc_report.json`
- 指定数据集配置文件：`--data_set /workspace/config/data_set.cfg`
- 输出日志到：`/workspace/logs/${MODEL_NAME}/test.log`

**验证执行结果**：
```bash
# 查看推理日志
tail -50 /workspace/logs/${MODEL_NAME}/test.log

# 检查结果文件
ls -lh /workspace/results/${MODEL_NAME}/predictions/
cat /workspace/results/${MODEL_NAME}/acc_report.json
```

### 关键性能指标

在每个数据集测评结束，日志会打印测评结果、指标和测试速度（单位：样本/秒），例如：
```text
结果: 1070/1070 成功
指标: {'accuracy': 0.9327102803738317, 'correct': 998, 'total': 1070, 'num_samples': 1070}
耗时: 56.7s
```

测试日志末尾会输出所有数据集结果汇总：

```text
============================================================
评测完成!
============================================================
评测数据集数: 1
  foundation-lid: {'accuracy': 0.9327102803738317, 'correct': 998, 'total': 1070, 'num_samples': 1070}, 样本数=1070

acc_report: /workspace/results/lang-id-voxlingua107-ecapa/acc_report.json
predictions: /workspace/results/lang-id-voxlingua107-ecapa/predictions/
```

其中，`acc_report` 记录了每个数据集的详细性能数据：

```json
{
  "records": [
    {
      "dataset": "foundation-lid",
      "model": "lang-id-voxlingua107-ecapa",
      "model_type": "understanding",
      "chip": "gpu",
      "device": "cuda (BW200)",
      "total_samples": 1000,
      "success_samples": 1000,
      "error_samples": 0,
      "success_rate": 1.0,
      "total_time": 66.92219638824463,
      "avg_inference_time": 0.04382076239585876,
      "metrics": {
        "accuracy": 0.653,
        "num_samples": 1000
      },
      "timestamp": "20260624_125504"
    }
  ],
  "last_updated": "20260624_125504",
  "notes": ""
}
```

#### 指标说明

| 类型 | 指标 | 说明 |
|------|------|------|
| 准确率（必采） | `accuracy` | 语种识别准确率，数值越高越好 |
| 性能（必采） | `avg_inference_time` | 平均每条样本推理时间，核心吞吐指标 |
| 准确率（辅助） | `success_rate` | 推理成功率 |
| 总样本数（辅助） | `total_samples` | 总共测试的样本数量 |

#### 指标采集

请严格使用下列代码进行指标采集，代码中必须将 acc_report.json 的内容读取后打印出来：

```
python3 -c "
import json
import sys
import os

result_path='/workspace/results/${MODEL_NAME}'
try:
    with open(os.path.join(result_path, 'acc_report.json'), 'r', encoding='utf-8') as f:
        acc = json.load(f)
    result = []
    for record in acc.get('records', []):
        metrics = record.get('metrics', {})

        result.append(
            {
                'dataset': record.get('dataset', ''),
                'avg_inference_time': record.get('avg_inference_time', 0),
                'success_rate': record.get('success_rate', 0),
                'total_samples': record.get('total_samples', 0),
                'accuracy': metrics.get('accuracy'),
            }
        )

    print(result)
    with open(os.path.join(result_path, 'result.json'), 'w', encoding='utf-8') as out:
        json.dump(result, out, indent=2, ensure_ascii=False)
    print('result.json written successfully.')
except Exception as e:
    result = {'status': 'error', 'message': str(e)}
    with open(os.path.join(result_path, 'result.json'), 'w', encoding='utf-8') as out:
        json.dump(result, out, indent=2, ensure_ascii=False)
    print(f'Error writing result.json: {e}', file=sys.stderr)
    sys.exit(1)
"
```

---

## 常见问题

1. **Docker 容器启动失败**
   - **容器名已存在**：确认不存在同名容器 `docker rm -f au-eval`
   - **设备文件不存在**：确认宿主机上有 `/dev/kfd`、`/dev/dri`、`/dev/mkfd` 设备文件
   - **Hygon HAL 库缺失**：确认 `/opt/hyhal` 目录存在且包含必要的库文件
   - **镜像未拉取**：确认镜像 `swr.cn-north-1.myhuaweicloud.com/deeplink/hygon-audio-au:latest` 已 pull 到本地
   - **共享内存不足**：如遇到内存错误，可增加 `--shm-size` 参数值（如 `256g`）

2. **容器内找不到 DCU 设备**
    - 验证设备挂载：`ls -lh /dev/kfd /dev/dri /dev/mkfd`
    - 检查驱动加载：`dmesg | grep -i hygon`（在宿主机执行）
    - 确认 HAL 库：`ls -lh /opt/hyhal/lib/`
    - 测试 DCU 可用性：在容器内运行 `rocm-smi` 或 `hy-smi`
    - 确认 docker run 时是否包含 `--device=/dev/kfd --device=/dev/dri --device=/dev/mkfd` 以及 `--group-add` 参数

3. **测试无法启动**
   - 检查模型路径 `/workspace/models/audio_understanding/lang-id-voxlingua107-ecapa/` 下 `embedding_model.ckpt`、`classifier.ckpt`、`hyperparams.yaml` 等关键文件是否存在
   - 检查数据集路径 `/workspace/datasets` 挂载是否正确
   - 检查配置文件 `/workspace/config/data_set.cfg` 是否存在且路径正确

4. **找不到模型或数据集**
   - 检查 `-v` 挂载参数是否正确映射宿主机路径到容器内路径
   - 确认挂载的宿主机目录存在且有读取权限

5. **SpeechBrain 模型加载失败 / 符号链接错误**
   - **现象**：报错提示无法在模型目录创建符号链接或临时文件
   - **原因**：SpeechBrain 在加载预训练模型时会尝试在模型目录创建符号链接缓存
   - **解决方案**：将 `AU_MODEL_CKPT_DIR` 对应的挂载改为 `:rw`（参考[容器创建命令](#容器创建命令)），确保模型目录可写

6. **DCU 显存不足**
   - **现象**：推理过程报错 `OutOfMemoryError`、`HIP error out of memory` 或程序卡死
   - **原因**：当前选中的 DCU 卡显存已被其他进程占用，或剩余显存不足以加载模型/数据
   - **解决方案**：
     step 1. **查看 DCU 使用情况**：
        ```bash
        rocm-smi
        # 或者
        hy-smi

        参考输出：
        ================================= System Management Interface ==================================
        ================================================================================================
        HCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      HCU%      Dec%      Enc%      Mode
        0       43.0C    131.0W     auto     800.0W     98%        0.0%      0.0%      0.0%      Normal
        1       42.0C    134.0W     auto     800.0W     99%        0.0%      0.0%      0.0%      Normal
        2       37.0C    131.0W     auto     800.0W     98%        0.0%      0.0%      0.0%      Normal
        3       37.0C    133.0W     auto     800.0W     100%       0.0%      0.0%      0.0%      Normal
        4       43.0C    133.0W     auto     800.0W     0%         0.0%      0.0%      0.0%      Normal
        5       42.0C    135.0W     auto     800.0W     0%         0.0%      0.0%      0.0%      Normal
        6       38.0C    133.0W     auto     800.0W     0%         0.0%      0.0%      0.0%      Normal
        7       36.0C    135.0W     auto     800.0W     0%         0.0%      0.0%      0.0%      Normal
        ================================================================================================
        ======================================== End of SMI Log ========================================
        ```
     step 2. **切换可用的 DCU 卡**：根据 `VRAM%` 的值，选取空闲的卡，通过 `HIP_VISIBLE_DEVICES` 环境变量指定：
        ```bash
        export HIP_VISIBLE_DEVICES=4  # 使用第4号卡
        # 或指定多张卡
        export HIP_VISIBLE_DEVICES=4,5,6,7
        ```
     step 3. **重新执行脚本**
   - **注意**：
     - `HIP_VISIBLE_DEVICES` 必须在启动 Python 脚本之前设置
     - 确保指定的 DCU ID 在容器中可见（可通过 `ls /dev/dri/card*` 或 `rocm-smi` 确认）

7. **日志或结果文件未生成**
   - 检查 `/workspace/logs` 和 `/workspace/results` 目录写权限
   - 确认挂载卷为 `:rw` 模式
