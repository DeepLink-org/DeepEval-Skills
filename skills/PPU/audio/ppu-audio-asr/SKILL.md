---
name: ppu-audio-asr
description: PPU 上语音识别模型推理性能评测技能。支持 SenseVoice 多语言语音识别模型，用于指导 executor 完成容器启动、数据集准备、推理执行、日志采集与性能指标分析（CER/WER）。
---

### 触发条件

当用户说以下任意内容时启动：
- "我要在 PPU 上跑 ASR 模型测试"
- "帮我测试 SenseVoice 多语言语音理解模型推理性能"
- "在 PPU 上跑 中文语音识别 Aishell-1 学术数据集测试集"
- "帮我批量测试语音识别模型 FP32/FP16 性能"
- "采集 ASR 模型 CER (字符错误率)"

---


### 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `ASR_PROJECT_ROOT` | `/workspace` | 否 | 项目根目录，需外部提供，模型文件所在目录 |
| `ASR_MODEL_CKPT_DIR` | `/workspace/models` | 是 | 模型检查点目录，存放speech_recognition场景下的模型权重文件 |
| `ASR_DATA_DIR` | `/workspace/datasets` | 是 | 包含speech_recognition场景下的数据集，具体包含Aishell-1 等数据集目录，存放 lmdb 格式数据 |
| `ASR_CONFIG_DIR` | `/workspace/config` | 是 | 配置文件目录，包含 data_set.cfg |
| `ASR_INFERENCE_OUTPUT` | `/workspace/results/${MODEL_NAME}` | 否 | 推理结果输出目录 |
| `ASR_LOGS_DIR` | `/workspace/logs/${MODEL_NAME}` | 否 | 日志输出目录 |

**说明**：
- **ASR_PROJECT_ROOT** 不需要外部提供，镜像中已包含所有推理代码和相关工具
- **ASR_MODEL_CKPT_DIR** 需要外部提供，存放 SenseVoiceSmall 模型权重文件
- **ASR_DATA_DIR** 需要外部提供，存放预处理后的数据集，包含speech_recognition场景下的数据集
- **ASR_CONFIG_DIR** 需要外部提供，存放 data_set.cfg 数据集路径配置文件
- 其他环境变量可根据需要自定义提供

**目录结构说明**：

- `$ASR_PROJECT_ROOT`: 项目根目录（`/workspace`），默认结构如下：
  ```
  $ASR_PROJECT_ROOT/                        # = /workspace
  ├── infer_runner.py                       # 推理主程序入口
  ├── inference.py                          # 推理核心逻辑
  ├── model_loader.py                       # 模型加载器
  ├── requirements.txt                      # Python 依赖清单
  ├── entrypoint.sh                         # 容器入口脚本
  ├── README.md                             # 项目说明文档
  ├── config/                               # = ASR_CONFIG_DIR
  │   ├── data_set.cfg                      # 数据集路径配置文件
  │   ├── global_env_state.json             # 全局环境状态
  │   ├── prometheus.yml                    # Prometheus 监控配置
  │   └── ...
  ├── models/
  │   └── speech_recognition/               # = ASR_MODEL_CKPT_DIR
  │       └── SenseVoiceSmall/              # ${MODEL_NAME}
  │           ├── README.md                 # 模型说明文档
  │           ├── model.pt                  # 模型权重文件
  │           ├── config.yaml               # 模型结构配置，定义编码器/解码器参数
  │           ├── configuration.json        # 模型推理配置 (JSON格式)
  │           ├── am.mvn                    # 声学模型均值方差归一化参数
  │           ├── tokens.json               # 词表文件，token ID 到文本的映射
  │           ├── chn_jpn_yue_eng_ko_spectok.bpe.model  # 多语言BPE分词模型 (中/日/粤/英/韩)
  │           ├── example/                  # 示例音频文件
  │           └── fig/                      # 模型架构示意图
  ├── datasets/                             # = ASR_DATA_DIR
  │   └── speech_recognition/               # 语音识别，总共 28 个数据集
  │       ├── aishell1/                     # Aishell-1 中文语音识别数据集
  │       │   ├── data_0.lmdb               # LMDB 格式数据分片 0
  │       │   ├── data_1.lmdb               # LMDB 格式数据分片 1
  │       │   └── meta.json                 # 数据集元信息
  │       ├── librispeech-test-clean/        # LibriSpeech test-clean 英文语音识别数据集
  │       ├── librispeech-test-other/        # LibriSpeech test-other 英文语音识别数据集
  │       ├── fleurs-cmn/                   # FLEURS 中文（普通话）语音识别数据集
  │       ├── fleurs-en/                    # FLEURS 英文语音识别数据集
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
- `model.pt`: 模型权重文件，推理的核心文件
- `config.yaml`: 模型结构配置，定义编码器/解码器参数
- `chn_jpn_yue_eng_ko_spectok.bpe.model`: 多语言 BPE 分词模型，支持中/日/粤/英/韩五语
- `tokens.json`: 词表映射文件
- `data_set.cfg`：数据集配置文件，指定各数据集 lmdb 路径
- 每个数据集目录下包含 `data_*.lmdb`（LMDB 数据分片）和 `meta.json`（元信息）

**注意**：
- 必需的参数（如 `ASR_MODEL_CKPT_DIR`、`ASR_DATA_DIR`、`ASR_CONFIG_DIR`）必须提供
- 表格中的"映射目录"列指明了容器启动时 `-v` 参数的挂载路径，即宿主机路径映射到容器内的路径

---

### 支持的模型配置

**当前支持模型**（共 1 个）：
- `SenseVoiceSmall`: 阿里达摩院发布的多语言语音识别模型，支持中/日/粤/英/韩五种语言的语音识别能力

**当前支持任务**：
- 基于语音识别数据集的语音识别模型推理与性能测试
- 性能评估：CER（字符错误率）、WER（词错误率）
- 吞吐性能：`avg_inference_time`（平均推理时间/样本）

**当前支持的数据集**：
语音识别任务总共支持 28 个数据集
aishell1、fleurs-ar、fleurs-cmn、fleurs-de、fleurs-en、fleurs-es、fleurs-fr、fleurs-hi、fleurs-it、fleurs-ja、fleurs-ko、fleurs-pt、fleurs-ru、fleurs-th、fleurs-vi、fleurs-yue、kespeech-beijing、kespeech-ji-lu、kespeech-jiang-huai、kespeech-jiao-liao、kespeech-lan-yin、kespeech-mandarin、kespeech-northeastern、kespeech-southwestern、kespeech-zhongyuan、librispeech-test-clean、librispeech-test-other、speech_asr_aishell1_testsets

**硬件要求**：
- 1 张 PPU 加速卡
- 足够显存支撑 ASR 模型推理

---

### 依赖要求

**容器镜像**：
```bash
open-audio-native-registry.cn-beijing.cr.aliyuncs.com/speech_recognition/sensevoice_small:ppu-v1.0.0
```

容器内已预装：
- Python 3.10
- PyTorch（适配 PPU 版本）
- PPU 运行时与工具链
- 模型推理运行器 `infer_runner.py`（位于 `/workspace/`）
- 相关 Python 依赖库

**宿主机依赖**：
- `/dev/alixpu*`：PPU 设备文件（必需，需通过 `--device` 挂载至容器内）

---

## 第一阶段：容器启动

### 容器创建命令

```bash
docker run -itd \
  --privileged=true \
  --ipc=host \
  --name asr-eval \
  $(for i in /dev/alixpu*; do [ -e "$i" ] && printf -- "--device=%s " "$i"; done) \
  -e PYTHONUNBUFFERED=1 \
  -v $ASR_CONFIG_DIR:/workspace/config:ro \
  -v $ASR_MODEL_CKPT_DIR:/workspace/models:ro \
  -v $ASR_DATA_DIR:/workspace/datasets:ro \
  -v $ASR_INFERENCE_OUTPUT:/workspace/results/${MODEL_NAME}:rw \
  -v $ASR_LOGS_DIR:/workspace/logs/${MODEL_NAME}:rw \
  open-audio-native-registry.cn-beijing.cr.aliyuncs.com/speech_recognition/sensevoice_small:ppu-v1.0.0
```

**公共参数说明**：

| 参数 | 说明 |
|------|------|
| `--privileged=true` | 以特权模式运行，允许访问 PPU 设备 |
| `--ipc=host` | 使用主机 IPC 命名空间，优化进程间通信 |
| `$(for i in /dev/alixpu*; ...)` | 动态发现并挂载所有 PPU 设备文件 |
| `-e PYTHONUNBUFFERED=1` | Python 输出不缓冲，确保日志实时可见 |

**挂载权限约定**：
- `:ro` — 只读，用于输入数据（模型权重、数据集、配置文件）
- `:rw` — 读写，用于输出目录（results、logs）

**注意**：
- 必须以 `-itd` 参数运行（交互式分离模式），保持容器后台运行
- 若已存在同名容器，先执行 `docker rm -f asr-eval`
- `${MODEL_NAME}` 在宿主机已通过 `export MODEL_NAME=SenseVoiceSmall` 设置

### 容器管理命令

**进入已创建的容器**：
```bash
# 如果容器在运行
docker exec -it asr-eval /bin/bash

# 如果容器已停止，先启动再进入
docker start asr-eval
docker exec -it asr-eval /bin/bash
```

**验证容器环境**：
```bash
# 检查 PPU 设备
ls -lh /dev/alixpu*

# 检查挂载的模型权重
ls -lh /workspace/models/speech_recognition/SenseVoiceSmall/model.pt

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
MODEL_NAME="SenseVoiceSmall"  # 可选: SenseVoiceSmall
DATASET_NAME="kespeech-beijing" # 可选：aishell1、kespeech-beijing等
cd /workspace
```

### 配置文件说明
**配置文件关键路径**（`/workspace/config/data_set.cfg`）：
```
# SenseVoiceSmall 数据集配置（容器内路径）
datasets_base: "/workspace/datasets"
datasets:
  - kespeech-beijing
```

### 步骤 2：执行推理评测

运行推理脚本，预测结果将保存至 `/workspace/results/${MODEL_NAME}`：

```bash
cd /workspace
mkdir -p /workspace/logs/${MODEL_NAME}
python3 -u /workspace/infer_runner.py \
  --model_dir /workspace/models/speech_recognition/${MODEL_NAME} \
  --output_dir /workspace/results/${MODEL_NAME}/predictions/ \
  --acc_report /workspace/results/${MODEL_NAME}/acc_report.json \
  --data_set /workspace/config/data_set.cfg \
  --dataset ${DATASET_NAME} 2>&1 | tee /workspace/logs/${MODEL_NAME}/test.log
```

上述指令的默认行为：
- 创建 `/workspace/logs/${MODEL_NAME}` 目录
- 启动 `/workspace/infer_runner.py` 进行推理
- 指定模型路径：`--model_dir /workspace/models/speech_recognition/${MODEL_NAME}`
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
结果: 265/265 成功
指标: {'cer': 0.10670493086355336, 'total_edits': 409, 'total_ref_len': 3833, 'substitutions': 347, 'deletions': 15, 'insertions': 47, 'num_samples': 265}
耗时: 23.4s
```

测试日志末尾会输出所有数据集结果汇总：

```text
============================================================
评测完成!
============================================================
评测数据集数: 1
  kespeech-beijing: {'cer': 0.10670493086355336, 'total_edits': 409, 'total_ref_len': 3833, 'substitutions': 347, 'deletions': 15, 'insertions': 47, 'num_samples': 265}, 样本数=265

acc_report: /workspace/results/SenseVoiceSmall/acc_report.json
predictions: /workspace/results/SenseVoiceSmall/predictions/
```

其中，`acc_report` 记录了每个数据集的详细性能数据：

```json
{
  "records": [
    {
      "dataset": "kespeech-beijing",
      "model": "SenseVoiceSmall",
      "model_type": "recognition",
      "chip": "ppu",
      "device": "ppu (PPU)",
      "total_samples": 265,
      "success_samples": 265,
      "error_samples": 0,
      "success_rate": 1.0,
      "total_time": 23.38645100593567,
      "avg_inference_time": 0.08438867924528304,
      "metrics": {
        "cer": 0.10670493086355336,
        "total_edits": 409,
        "total_ref_len": 3833,
        "substitutions": 347,
        "deletions": 15,
        "insertions": 47,
        "num_samples": 265
      },
      "timestamp": "20260602_030457"
    }
  ],
  "last_updated": "20260602_030457",
  "notes": ""
}
```

#### 指标说明

| 类型 | 指标 | 说明 |
|------|------|------|
| 准确率（必采） | CER | 字符错误率，数值越低越好（中文数据集） |
| 准确率（必采） | WER | 词错误率，数值越低越好（英文数据集） |
| 性能（必采） | `avg_inference_time` | 平均每条样本推理时间，核心吞吐指标 |
| 准确率（辅助） | `success_rate` | 推理成功率 |
| 总样本数（辅助） | `total_samples` | 总共测试的样本数量 |

**注意**：中文数据集仅有 CER 指标；英文数据集仅有 WER 指标

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
                'cer': metrics.get('cer'),
                'wer': metrics.get('wer'),
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

1. **容器启动失败**
   - **容器名已存在**：确认不存在同名容器 `docker rm -f asr-eval`
   - **设备文件不存在**：确认宿主机上有 `/dev/alixpu*` 设备文件（`ls /dev/alixpu*`）
   - **镜像未拉取**：确认镜像 `open-audio-native-registry.cn-beijing.cr.aliyuncs.com/speech_recognition/sensevoice_small:ppu-v1.0.0` 已 pull 到本地
   - **特权权限不足**：确保 `--privileged=true` 参数已设置

2. **容器内找不到 PPU 设备**
    - 验证设备挂载：`ls -lh /dev/alixpu*`
    - 确认宿主机上 PPU 驱动已正确加载
    - 确认 `docker run` 时已通过 `--device=/dev/alixpu*` 动态挂载所有 PPU 设备
    - 检查 `--privileged=true` 与 `--ipc=host` 是否生效

3. **测试无法启动**
   - 检查模型路径 `/workspace/models/speech_recognition/SenseVoiceSmall/` 下 `model.pt` 权重文件是否存在
   - 检查数据集路径 `/workspace/datasets` 挂载是否正确
   - 检查配置文件 `/workspace/config/data_set.cfg` 是否存在且路径正确

4. **找不到模型或数据集**
   - 检查 `-v` 挂载参数是否正确映射宿主机路径到容器内路径
   - 确认挂载的宿主机目录存在且有读取权限

5. **PPU 显存不足**
   - **现象**：推理过程报错显存不足（`Out of Memory`）或程序卡死
   - **原因**：当前选中的 PPU 卡显存已被其他进程占用，或剩余显存不足以加载模型/数据
   - **解决方案**：
     step 1. **查看 PPU 使用情况**：在宿主机或容器内查询 PPU 状态，确认哪些卡处于空闲
     step 2. **切换可用的 PPU 卡**：根据空闲情况，通过环境变量指定可见设备（具体环境变量名以 PPU 运行时文档为准），例如：
        ```bash
        export ALIXPU_VISIBLE_DEVICES=0  # 使用第0号卡
        ```
     step 3. **重新执行脚本**
   - **注意**：可见设备环境变量必须在启动 Python 脚本之前设置

6. **日志或结果文件未生成**
   - 检查 `/workspace/logs` 和 `/workspace/results` 目录写权限
   - 确认挂载卷为 `:rw` 模式
