---
name: hygon-science-weather
description: Hygon DCU 上全球中期天气预报模型推理与性能评测技能。支持 FengWu、FourCastNet、FuXi、GraphCast、Pangu-Weather 等主流气象大模型，用于指导 executor 完成容器启动、数据集准备、推理执行、日志采集与性能指标分析（RMSE/ACC）。
---

### 触发条件

当用户说以下任意内容时启动：
- "我要在 Hygon 上跑天气预测模型推理测试"
- "帮我测试 FengWu / FourCastNet / FuXi 推理性能"
- "在 Hygon DCU 上跑 weather forecasting inference benchmark"
- "帮我批量测试气象模型 RMSE/ACC 指标"
- "采集 weather model 推理性能"

---


### 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `WEATHER_PROJECT_ROOT` | `/workspace/model` | 是 | 项目根目录，需外部提供，模型文件所在目录 |
| `WEATHER_MODEL_CKPT_DIR` | `/workspace/model/${MODEL_NAME}/data/checkpoints` | 是 | 模型检查点目录，存放模型ckpt文件 |
| `WEATHER_DATA_DIR` | `/workspace/model/${MODEL_NAME}/data/data` | 否 | 预处理后的ERA5数据文件目录，存放h5格式数据 |
| `WEATHER_STATS_DIR` | `/workspace/model/${MODEL_NAME}/data/stats/` | 否 | 预处理后的统计量文件目录，包含均值/标准差等统计信息 |
| `WEATHER_STATIC_DIR` | `/workspace/model/${MODEL_NAME}/data/static/` | 否 | 预处理后的静态特征目录，包含陆地掩码等静态数据 |
| `WEATHER_DATA_ORIGIN` | `/workspace/model/${MODEL_NAME}/data/data/nc` | 否 | 原始NC文件目录，存放原始的nc文件 |
| `WEATHER_DUMMY_DATA_DIR` | `/workspace/model/${MODEL_NAME}/dummy_data` | 否 | 虚拟数据目录，用于快速验证流程 |
| `WEATHER_INFERENCE_OUTPUT` | `/workspace/results/${MODEL_NAME}` | 否 | 推理结果输出目录 |
| `WEATHER_LOGS_DIR` | `/workspace/logs/${MODEL_NAME}` | 否 | 日志输出目录 |

**说明**：
- **WEATHER_PROJECT_ROOT** 需要外部提供，包含所有模型代码和相关工具
- **WEATHER_MODEL_CKPT_DIR** 存放模型ckpt文件，即模型检查点
- **WEATHER_DATA_DIR**、**WEATHER_STATS_DIR**、**WEATHER_STATIC_DIR** 存放预处理后的数据集，分别对应数据、统计数据和静态特征
- **WEATHER_DATA_ORIGIN** 存放原始的nc文件
- **WEATHER_DUMMY_DATA_DIR** 存放虚拟数据，用于快速验证流程，包含 dummy_data/data、dummy_data/stats、dummy_data/static 子目录
- 数据预处理过程中会生成中间目录：`$WEATHER_DATA_DIR/nc`（原始数据）、`$WEATHER_DATA_DIR/tmp_h5`（临时数据）、`$WEATHER_DATA_DIR/h5`（合并后数据）等
- 其他环境变量可根据需要自定义提供

**目录结构说明**：

- `$WEATHER_PROJECT_ROOT`: 项目根目录，模型文件所在目录，默认结构如下：
  ```
  $WEATHER_PROJECT_ROOT/
  ├── era5_dataset_prepare/    # ERA5 数据预处理工具
  │   ├── step_1_data_download.py     # 下载ERA5原始数据
  │   ├── step_2_data_conversion.py   # 数据格式转换
  │   ├── step_3_data_merge.py        # 年度数据合并
  │   └── step_4_stats_calculate.py   # 统计量计算
  ├── fengwu/                  # FengWu 模型
  ├── fourcastnet/             # FourCastNet 模型
  ├── fuxi/                    # FuXi 模型
  ├── graphcast/               # GraphCast 模型
  └── pangu_weather/           # Pangu-Weather 模型
  ```

  每个模型目录的典型结构：
  ```
  <model_name>/
  ├── conf/                    # 配置文件目录
  │   └── config.yaml          # 核心配置文件（定义相对路径）
  ├── data/                    # 数据和检查点目录（可通过环境变量自定义挂载位置）
  │   ├── data/                # ERA5 数据文件 ({year}.h5) → WEATHER_DATA_DIR
  │   │   ├── nc/              # 原始 ERA5 NC 数据文件 (.nc) → WEATHER_DATA_ORIGIN
  │   │   ├── h5/              # 合并后的年度H5数据文件 ({year}.h5) → 数据预处理产物
  │   │   └── tmp_h5/          # 数据转换过程中的临时H5文件 → 数据预处理中间产物
  │   ├── stats/               # 统计量文件 → WEATHER_STATS_DIR
  │   ├── static/              # 静态特征 → WEATHER_STATIC_DIR
  │   └── checkpoints/         # 模型权重文件 → WEATHER_MODEL_CKPT_DIR
  ├── dummy_data/              # 虚拟数据目录 → WEATHER_DUMMY_DATA_DIR
  │   ├── data/                # 虚拟 ERA5 数据文件 (.h5)
  │   ├── stats/               # 虚拟统计量文件 (.npy)
  │   └── static/              # 虚拟静态特征文件
  ├── result/                  # 推理和评估结果
  ├── inference.py             # 推理脚本
  ├── result.py                # 评估脚本
  └── train.py                 # 训练脚本（如需要）
  ```

  

**注意**：
- 必需的参数（如 `WEATHER_PROJECT_ROOT`、`WEATHER_MODEL_CKPT_DIR`）必须提供
- `WEATHER_DATA_DIR`、`WEATHER_STATS_DIR`、`WEATHER_STATIC_DIR` 需要同时提供预处理数据，且优先级高于 `WEATHER_DATA_ORIGIN`
- `WEATHER_DATA_ORIGIN` 为可选参数，仅在预处理数据不存在时使用，指向原始NC文件目录，按照`era5_dataset_prepare`中的步骤预处理数据
- 当 `WEATHER_DATA_DIR`、`WEATHER_STATS_DIR`、`WEATHER_STATIC_DIR`、`WEATHER_DATA_ORIGIN` 均不存在或为空时，可以使用 `WEATHER_DUMMY_DATA_DIR` 生成虚拟数据进行快速验证
- 表格中的"映射目录"列指明了容器启动时 `-v` 参数的挂载路径，即宿主机路径映射到容器内的路径


---


### 支持的模型配置

**当前支持模型**（共 5 个）：
- `fengwu`: 上海人工智能实验室联合多所高校发布的全球中期天气预报大模型，基于多模态和多任务深度学习方法，首次实现在高分辨率上对核心大气变量超过 10 天的有效预报
- `fourcastnet`: 基于 AFNO 的高分辨率全球天气预报模型
- `fuxi`: 复旦大学发布的多尺度气象预测模型
- `graphcast`: DeepMind 发布的基于图神经网络的全球天气预报模型
- `pangu_weather`: 华为云发布的盘古气象大模型

**当前支持任务**：
- 基于 ERA5 数据集的气象模型推理与预测
- 性能评估：RMSE（均方根误差）、ACC（异常相关系数）
- 可视化：Loss 曲线、预测场对比图

**硬件要求**：
- 1 张 Hygon DCU GPU（推理阶段）

---

### 依赖要求

**Docker 镜像**：
```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/hygon-science-weather:latest
```

容器内已预装：
- PyTorch 2.5.1
- Hygon DTK 25.04.2（DCU 工具包）
- Python 3.10
- onescience 框架
- numpy, h5py, matplotlib, tqdm, pyyaml 等


**数据预处理工具**（位于 `$WEATHER_PROJECT_ROOT/era5_dataset_prepare/`）：
- `step_1_data_download.py`: 下载 ERA5 原始数据
- `step_2_data_conversion.py`: 数据格式转换
- `step_3_data_merge.py`: 年度数据合并
- `step_4_stats_calculate.py`: 统计量（均值/标准差）计算

---


## 第一阶段：容器启动

### 选择模型与数据场景

启动容器前，先指定目标模型，再按优先级判断数据场景，选择对应的启动命令：

```bash
# 1. 选择要测试的模型
export MODEL_NAME="fengwu"  # 可选: fengwu, fourcastnet, fuxi, graphcast, pangu_weather

# 2. 判断数据场景
if [ -n "$WEATHER_DATA_DIR" ] && [ -n "$WEATHER_STATS_DIR" ] && [ -n "$WEATHER_STATIC_DIR" ]; then
    # 场景 A：已有预处理数据（优先使用）
    DATA_SCENARIO="preprocessed"
elif [ -n "$WEATHER_DATA_ORIGIN" ]; then
    # 场景 B：从原始 NC 数据开始处理
    DATA_SCENARIO="raw"
elif [ -n "$WEATHER_DUMMY_DATA_DIR" ]; then
    # 场景 C：虚拟数据快速验证
    DATA_SCENARIO="dummy"
else
    echo "ERROR: 必须提供 WEATHER_DATA_DIR / WEATHER_DATA_ORIGIN / WEATHER_DUMMY_DATA_DIR 至少一个"
    exit 1
fi
```

### 容器创建命令

**挂载权限约定**：
- `:ro` — 只读，用于输入数据（checkpoint、数据集、HAL 库等），防止误修改
- `:rw` — 读写，用于输出目录（results、logs）和需要写入的目录

**公共参数**（所有场景共享）：

| 参数 | 说明 |
|------|------|
| `--network=host --ipc=host` | 使用主机网络和 IPC，优化进程间通信 |
| `--shm-size=16G` | 共享内存大小，避免大数据加载时内存不足 |
| `--device=/dev/kfd --device=/dev/mkfd --device=/dev/dri` | 挂载 Hygon DCU 设备文件 |
| `-v /opt/hyhal:/opt/hyhal:ro` | 挂载 Hygon HAL 库（必需，只读） |
| `--group-add video --cap-add=SYS_PTRACE` | 添加 GPU 访问权限与调试能力 |
| `--security-opt seccomp=unconfined` | 禁用 seccomp 安全策略 |

**公共卷挂载**（所有场景必需）：
```bash
-v $WEATHER_PROJECT_ROOT:/workspace/model:rw \
-v $WEATHER_MODEL_CKPT_DIR:/workspace/model/${MODEL_NAME}/data/checkpoints:ro \
-v $WEATHER_INFERENCE_OUTPUT:/workspace/results/${MODEL_NAME}:rw \
-v $WEATHER_LOGS_DIR:/workspace/logs/${MODEL_NAME}:rw
```

---

**场景 A：已有预处理数据**

```bash
docker run -it \
  --name weather_inference \
  --network=host --ipc=host --shm-size=16G \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v $WEATHER_PROJECT_ROOT:/workspace/model:rw \
  -v $WEATHER_MODEL_CKPT_DIR:/workspace/model/${MODEL_NAME}/data/checkpoints:ro \
  -v $WEATHER_DATA_DIR:/workspace/model/${MODEL_NAME}/data/data:ro \
  -v $WEATHER_STATS_DIR:/workspace/model/${MODEL_NAME}/data/stats:ro \
  -v $WEATHER_STATIC_DIR:/workspace/model/${MODEL_NAME}/data/static:ro \
  -v $WEATHER_INFERENCE_OUTPUT:/workspace/results/${MODEL_NAME}:rw \
  -v $WEATHER_LOGS_DIR:/workspace/logs/${MODEL_NAME}:rw \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  swr.cn-north-1.myhuaweicloud.com/deeplink/hygon-science-weather:latest \
  /bin/bash
```

**场景 B：从原始 NC 数据开始**（挂载 `WEATHER_DATA_ORIGIN` 代替数据/统计/静态目录）

```bash
docker run -it \
  --name weather_inference \
  --network=host --ipc=host --shm-size=16G \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v $WEATHER_PROJECT_ROOT:/workspace/model:rw \
  -v $WEATHER_MODEL_CKPT_DIR:/workspace/model/${MODEL_NAME}/data/checkpoints:ro \
  -v $WEATHER_DATA_ORIGIN:/workspace/model/${MODEL_NAME}/data/data/nc:ro \
  -v $WEATHER_INFERENCE_OUTPUT:/workspace/results/${MODEL_NAME}:rw \
  -v $WEATHER_LOGS_DIR:/workspace/logs/${MODEL_NAME}:rw \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  swr.cn-north-1.myhuaweicloud.com/deeplink/hygon-science-weather:latest \
  /bin/bash
```

**场景 C：虚拟数据快速验证**

```bash
docker run -it \
  --name weather_inference \
  --network=host --ipc=host --shm-size=16G \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v $WEATHER_PROJECT_ROOT:/workspace/model:rw \
  -v $WEATHER_MODEL_CKPT_DIR:/workspace/model/${MODEL_NAME}/data/checkpoints:ro \
  -v $WEATHER_DUMMY_DATA_DIR:/workspace/model/${MODEL_NAME}/dummy_data:rw \
  -v $WEATHER_INFERENCE_OUTPUT:/workspace/results/${MODEL_NAME}:rw \
  -v $WEATHER_LOGS_DIR:/workspace/logs/${MODEL_NAME}:rw \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  swr.cn-north-1.myhuaweicloud.com/deeplink/hygon-science-weather:latest \
  /bin/bash
```

**注意**：
- 所有大文件路径通过环境变量自定义，可存放在大容量磁盘上
- 若已存在同名容器，先执行 `docker rm -f weather_inference`
- `${MODEL_NAME}` 在宿主机已通过 `export MODEL_NAME=...` 设置

### 容器管理命令

**进入已创建的容器**：
```
# 如果容器已在运行
docker exec -it weather_inference /bin/bash

# 如果容器已停止，先启动再进入
docker start weather_inference
docker exec -it weather_inference /bin/bash
```

**验证容器环境**：
```
# 设置模型名称（与宿主机 docker run 时一致）
MODEL_NAME="fengwu"  # 可选: fengwu, fourcastnet, fuxi, graphcast, pangu_weather

# 检查 DCU 设备
ls -lh /dev/kfd /dev/mkfd /dev/dri

# 检查 HAL 库
ls -lh /opt/hyhal/lib/

# 检查挂载的目录
ls -lh /workspace/model/
ls -lh /workspace/model/${MODEL_NAME}/
ls -lh /workspace/results/${MODEL_NAME}/
ls -lh /workspace/logs/${MODEL_NAME}/
```
---



## 第二阶段：容器中执行评测

### 步骤 1：进入模型目录

容器内所有路径已通过卷挂载固定（详见[环境变量定义](#环境变量定义)），无需额外设置环境变量。

```bash
# 选择要测试的模型
MODEL_NAME="fengwu"  # 可选: fengwu, fourcastnet, fuxi, graphcast, pangu_weather
cd /workspace/model/$MODEL_NAME
```

### 配置文件说明

**配置文件关键路径**（`conf/config.yaml`）：
  ```yaml
  stats_dir: 均值/标准差文件路径，用于归一化
  static_dir: 静态文件路径（陆地掩码等），若模型不需要可忽略
  data_dir: ERA5 数据根路径，年度 h5 文件存放于 data_dir/data/{year}.h5
  train_time: [2000, 2001]   # 训练年份
  val_time: [2002]            # 验证年份
  test_time: [2003]           # 测试年份
  ```
  
  **注意**：这些路径均为相对路径，相对于模型目录 `<model_name>/`。容器内实际路径为：
  - `checkpoint_dir`: `/workspace/model/${MODEL_NAME}/data/checkpoints`
  - `stats_dir`: `/workspace/model/${MODEL_NAME}/data/stats/`
  - `static_dir`: `/workspace/model/${MODEL_NAME}/data/static/`
  - `data_dir`: `/workspace/model/${MODEL_NAME}/data/`

### 步骤 2：数据集准备（如需要）

**提醒**：执行数据准备前，需根据实际数据场景适配 `conf/config.yaml` 中的路径配置，确保 `data_dir`、`stats_dir`、`static_dir` 指向正确的数据目录。

**场景 A：已有处理好的 ERA5 数据**

如果 `/workspace/model/${MODEL_NAME}/data/data/` 目录下已有 `{year}.h5` 文件，且 `/workspace/model/${MODEL_NAME}/data/stats/` 目录下有 `global_means.npy` 和 `global_stds.npy`，可跳过此步骤。

**验证数据就绪**：
```bash
# 检查数据文件
ls -lh /workspace/model/${MODEL_NAME}/data/data/*.h5
ls -lh /workspace/model/${MODEL_NAME}/data/stats/*.npy

# 检查配置文件中的路径设置
cat /workspace/model/$MODEL_NAME/conf/config.yaml | grep -A 3 "dataset:"
```

**场景 B：需要从原始数据开始处理**

如果 `/workspace/model/${MODEL_NAME}/data/data/nc/` 目录下有原始的 ERA5 NC 文件，需要执行以下步骤进行数据预处理：

```bash
# 1. 进入数据目录
cd /workspace/model/${MODEL_NAME}/data/data

# 2. 从预处理工具目录复制脚本到数据目录
cp /workspace/model/era5_dataset_prepare/*.py .

# 3. 执行数据预处理流程
python step_2_data_conversion.py  # 转换数据格式到 ./tmp_h5 目录
python step_3_data_merge.py  # 合并数据到 ./h5 目录
python step_4_stats_calculate.py  # 计算统计量到 /workspace/model/${MODEL_NAME}/data/stats 目录
```

数据预处理完成后，数据会直接生成在正确的目录中：
- `/workspace/model/${MODEL_NAME}/data/data/h5/` 目录下会有 `{year}.h5` 文件
- `/workspace/model/${MODEL_NAME}/data/stats/` 目录下会有统计量文件

**注意**：
- 如果使用自定义的大容量存储路径，建议直接移动文件而非创建符号链接
- 确保 `conf/config.yaml` 中的相对路径能正确映射到容器内的挂载路径

**场景 C：使用虚拟数据进行快速验证**

当 `WEATHER_DATA_DIR`、`WEATHER_STATS_DIR`、`WEATHER_STATIC_DIR`、`WEATHER_DATA_ORIGIN` 均不存在或为空时，可以使用 `WEATHER_DUMMY_DATA_DIR` 生成虚拟数据进行快速验证流程。

```bash
# 1. 检查虚拟数据目录是否存在
ls -lh /workspace/model/${MODEL_NAME}/dummy_data/

# 2. 如果虚拟数据目录存在，创建符号链接
ln -s /workspace/model/${MODEL_NAME}/dummy_data/data /workspace/model/$MODEL_NAME/data/data
ln -s /workspace/model/${MODEL_NAME}/dummy_data/stats /workspace/model/$MODEL_NAME/data/stats
ln -s /workspace/model/${MODEL_NAME}/dummy_data/static /workspace/model/$MODEL_NAME/data/static

# 3. 验证虚拟数据
ls -lh /workspace/model/$MODEL_NAME/data/data/*.h5
ls -lh /workspace/model/$MODEL_NAME/data/stats/*.npy
```

**注意**：
- 虚拟数据仅用于快速验证流程，不适用于实际性能评估
- 虚拟数据通常包含少量的样本数据，用于测试代码逻辑
- 如果需要生成虚拟数据，可以运行 `python fake_data.py` 脚本

### 步骤 3：执行推理

运行推理脚本，预测结果将保存至模型目录下的 `./result/output/` ：

```
cd /workspace/model/$MODEL_NAME

# 执行推理（结果输出到 ./result/output/）
python inference.py 2>&1 | tee /workspace/logs/${MODEL_NAME}_inference.log
```

**推理输出**：
- 文件格式：`YYYYMMDDHH.npy`（例如：`1954010106.npy`）
- 数据形状：`[C, H, W]`（通道数 × 纬度 × 经度）
- 输出位置：`/workspace/model/${MODEL_NAME}/result/output/`
- 默认测试时间由 `conf/config.yaml` 中的 `test_time` 指定

**注意**：
- 如需快速验证流程，可先运行 `python fake_data.py` 生成虚拟数据（需将conf/config.yaml中max_epoch设为1）

**验证推理结果**：
```
# 检查输出文件
ls -lh /workspace/model/${MODEL_NAME}/result/output/*.npy

# 查看推理日志
tail -50 /workspace/logs/${MODEL_NAME}_inference.log
```

### 关键性能指标

#### 执行评估

```
cd /workspace/model/$MODEL_NAME

# 执行评估（计算 RMSE/ACC，生成可视化图表）
python result.py 2>&1 | tee /workspace/logs/${MODEL_NAME}_eval.log
```

**评估输出产物**（位于 `/workspace/model/${MODEL_NAME}/result/`）：

| 文件路径 | 描述 |
| :--- | :--- |
| `rmse.npy` | 各通道均方根误差 (RMSE) |
| `acc.npy` | 各通道异常相关系数 (ACC) |
| `loss.png` | 训练与验证损失曲线图 |
| `{date}_{var}.png` | 特定时间步的真值、预测值及差异对比图 |

#### 指标说明

| 类型 | 指标 | 说明 |
|------|------|------|
| 精度（必采） | RMSE | 均方根误差，数值越低越好 |
| 精度（必采） | ACC | 异常相关系数，数值越高越好 |
| 可视化（辅助） | loss.png | 训练与验证损失曲线 |
| 可视化（辅助） | 预测对比图 | 真值与预测值差异对比 |

**控制台输出示例**：
```
┌──────────────────────┬──────────────┬──────────────┐
│ Channel              │         RMSE │          ACC │
├──────────────────────┼──────────────┼──────────────┤
│ 2m_temperature       │       1.2345 │       0.9876 │
│ geopotential_500     │      12.3456 │       0.9543 │
│ ...                  │          ... │          ... │
├──────────────────────┼──────────────┼──────────────┤
│ Average              │       5.6789 │       0.9700 │
└──────────────────────┴──────────────┴──────────────┘
```

#### 指标采集

必须使用以下脚本统一采集指标并输出到 `result.json`，**禁止**通过 grep / 日志解析等不可靠方式提取指标：

```
cd /workspace/model/${MODEL_NAME}
python -c "
import numpy as np, json, sys, os

rmse = np.load('result/rmse.npy')
acc = np.load('result/acc.npy')

metrics = {
    'average_rmse': float(np.mean(rmse)),
    'average_acc': float(np.mean(acc)),
    'per_channel_rmse': rmse.tolist(),
    'per_channel_acc': acc.tolist()
}

result = {
    'status': 'success',
    'model_name': os.environ.get('MODEL_NAME', 'unknown'),
    'metrics': metrics
}

with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f'Average RMSE: {metrics[\"average_rmse\"]:.6f}')
print(f'Average ACC:  {metrics[\"average_acc\"]:.6f}')
print('Metrics saved to result.json')
"
if [ $? -ne 0 ]; then
    echo "ERROR: Metrics collection failed!" >&2
    exit 1
fi

echo "===== Metrics collection completed ====="
```

**约束规则**：
- **必须**从 `.npy` 文件加载指标，**禁止**从控制台日志 grep/awk 提取
- **必须**输出到 `result.json`，格式固定为 `{status, model_name, metrics}`
- **必须**校验脚本退出码，失败时 `exit 1` 阻断后续流程


#### 自定义可视化（可选）

修改 `result.py` 末尾的参数指定评估日期和变量：

```
# 编辑 result.py
vim /workspace/model/$MODEL_NAME/result.py

# 修改以下变量（在 if __name__ == "__main__": 块中）
# test_year = cfg_data.dataset.test_time[0]  # 选择测试年份
# eg_files = ['1954010106']                   # 指定具体日期（YYYYMMDDHH 格式）
# channel_index = [2, 5, 10]                  # 选择要可视化的变量索引

# 重新运行评估
python result.py
```

常用变量索引参考：
- `2m_temperature`: 索引 2
- `geopotential_500`: 索引约 27（具体需查看 config.yaml 中的 channels 列表）
- `temperature_500`: 需要根据实际配置确认

---

## 常见问题

1. **Docker 容器启动失败**
    - **设备文件不存在**：确认宿主机上有 `/dev/kfd`、`/dev/mkfd`、`/dev/dri` 设备文件
    - **Hygon HAL 库缺失**：确认 `/opt/hyhal` 目录存在且包含必要的库文件
    - **权限问题**：确保当前用户有访问 DCU 设备的权限，或已加入 `video` 组
    - **共享内存不足**：如遇到内存错误，可增加 `--shm-size` 参数值（如 `32G`）
    - **端口冲突**：使用 `--network=host` 时确保主机上没有占用所需端口

2. **容器内找不到 DCU 设备**
    - 验证设备挂载：`ls -lh /dev/kfd /dev/mkfd /dev/dri`
    - 检查驱动加载：`dmesg | grep -i hygon`（在宿主机执行）
    - 确认 HAL 库：`ls -lh /opt/hyhal/lib/`
    - 测试 DCU 可用性：在容器内运行 `rocminfo` 或 `hipinfo`（如可用）

3. **容器内路径与配置不匹配**
    - 容器内模型路径为 `/workspace/model/<model_name>`，需相应调整工作目录
    - 推理结果输出到 `/workspace/model/${MODEL_NAME}/result/`，评估产物（rmse.npy、acc.npy 等）位于 `result/` 子目录下
    - 容器内路径已通过卷挂载固定，无需设置环境变量（见[环境变量定义](#环境变量定义)）
4. **需要切换 DCU 卡**
    通过设置 `HIP_VISIBLE_DEVICES` 环境变量指定可用的 GPU ID。
    例如，如果卡 0-3 被占用，可以尝试使用卡 4：
    ```bash
    export HIP_VISIBLE_DEVICES=4
    ```
    或者指定多张卡：
    ```bash
    export HIP_VISIBLE_DEVICES=4,5,6,7
    ```
4. **GPU 显存不足**
    - **现象**：推理或评估过程中报错 `OutOfMemoryError`、`HIP error out of memory` 或程序卡死。
    - **原因**：当前选中的 DCU 卡显存已被其他进程占用，或剩余显存不足以加载模型/数据。
    - **解决方案**：
      step 1. **查看 DCU 使用情况**：
         ```bash
         # 在容器内或宿主机执行
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
      step 2. **切换可用的 DCU 卡**
      根据`VRAM%`的值，选取空闲的卡，使用`HIP_VISIBLE_DEVICES`环境变量指定。
      step 3. **重新执行脚本**：
    - **注意**：
      - `HIP_VISIBLE_DEVICES` 必须在启动 Python 脚本之前设置。
      - 确保指定的 GPU ID 在容器中可见（可通过 `ls /dev/dri/card*` 或 `rocm-smi` 确认）。
      - 如果是多卡并行推理，请确保代码支持分布式或多卡配置。
---
