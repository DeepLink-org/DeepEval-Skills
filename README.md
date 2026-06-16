# DeepEval-Skills

人工智能软硬件验证平台 评测 Skills

本仓库遵循 [Agent Skills 标准](https://agentskills.io/specification)。


## 快速开始

DeepEval-Skills 兼容 **Claude Code**、**Cursor**、**Codex** 以及任何支持 [Agent Skills 标准](https://agentskills.io/specification) 的智能体。也可支持Deeplink自研的评测智能体**DeepEval**（已内置**DeepEval-Skills**）进行开箱即用评测。

### npx（适用于所有智能体）

使用 [`skills`](https://www.npmjs.com/package/skills) CLI 直接安装：

```bash
# 克隆本仓库到本地
git clone <repo-url> DeepEval-Skills

# 查看本仓库中可用的 skills
npx skills add ./DeepEval-Skills --list

# 批量安装 skill 到当前项目
npx skills add ./DeepEval-Skills/skills/NVIDIA -s '*'

# 安装指定 skill 到当前项目
npx skills add ./DeepEval-Skills/skills/NVIDIA/nlp/nvidia-nlp-inference

# 仅安装到指定智能体
npx skills add ./DeepEval-Skills/skills/NVIDIA -s '*' --agent claude-code
```

claude-code 使用示例
```
> 我要评测NVIDIA上GEMM算子的性能
> [Skill自动加载] nvidia-nlp-operator 
> [开始自动化评测流程]

```


## Skill 导航

下表列出当前可用的评测技能。

| **场景** | **子场景** | **芯片** | **Skills** | **说明** |
|---------|----------|--------|----------|---------|
| 语言场景 | 训练 | NVIDIA | [nvidia-nlp-training](skills/NVIDIA/nlp/nvidia-nlp-training) ✅ | 支持大规模语言模型预训练，包括模型初始化、数据加载、分布式训练、梯度同步等全流程性能评测。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 微调 | NVIDIA | [nvidia-nlp-finetune](skills/NVIDIA/nlp/nvidia-nlp-finetune) ✅ | 实现语言模型指令微调、领域适配等任务的性能评测，涵盖LoRA、QLoRA等高效微调技术。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 推理 | NVIDIA | [nvidia-nlp-inference](skills/NVIDIA/nlp/nvidia-nlp-inference) ✅ | 实现语言模型在线推理性能评测，包括吞吐量、延迟、显存占用等关键指标。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 算子 | NVIDIA | [nvidia-nlp-operator](skills/NVIDIA/nlp/nvidia-nlp-operator) ✅ | 提供常见NLP基础算子（GEMM、Attention、FFN、LayerNorm等）性能测试，支持不同精度、批量大小、序列长度的组合测试。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| 视觉场景 | 检测训练 | NVIDIA | [nvidia-cv-detection](skills/NVIDIA/cv/nvidia-cv-detection) ✅ | 实现目标检测模型（如YOLO、Faster R-CNN）训练性能评测，包括数据增强、损失计算、后处理等环节。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 分类训练 | NVIDIA | [nvidia-cv-pretrain](skills/NVIDIA/cv/nvidia-cv-pretrain) ✅ | 实现图像分类模型（如ResNet、Vision Transformer）训练性能评测，涵盖图像预处理、模型前向/反向传播等。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 分割训练 | NVIDIA | [nvidia-cv-segmentation](skills/NVIDIA/cv/nvidia-cv-segmentation) ✅ | 实现图像分割模型（如U-Net、DeepLab）训练性能评测，包括像素级标注、分割掩码生成等任务。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 视觉感知推理 | NVIDIA | ⏳ 敬请期待 | 计划实现视觉感知任务性能评测，包括图像识别、物体检测、场景理解等综合视觉能力评估。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| 多模态场景 | 文生图推理 | NVIDIA | [nvidia-mm-t2i](skills/NVIDIA/mm/nvidia-mm-t2i) ✅ | 实现文本到图像生成模型（如Stable Diffusion、DALL-E）推理性能评测，包括提示词编码、扩散过程、图像解码等阶段。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 文生视频推理 | NVIDIA | [nvidia-mm-t2v](skills/NVIDIA/mm/nvidia-mm-t2v) ✅ | 实现文本到视频生成模型推理性能评测，涵盖时序建模、帧间一致性、视频质量评估等指标。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 多模态理解推理 | NVIDIA | ⏳ 敬请期待 | 计划实现多模态理解任务性能评测，包括图文理解、跨模态检索、视觉问答等综合能力评估。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| 科学计算场景 | 材料科学（推理） | NVIDIA | ⏳ 敬请期待 | 计划实现材料科学研究任务性能评测，包括分子模拟、材料特性预测、晶体结构分析等科学计算应用。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 气象科学（推理） | NVIDIA | ⏳ 敬请期待 | 计划实现气象模拟任务性能评测，包括天气预报模型、气候模拟、大气环流计算等科学计算应用。 |
| | | 海光 DCU | [hygon-science-weather](skills/Hygon/science/hygon-science-weather) ✅ | 支持 FengWu、FourCastNet、FuXi、GraphCast、Pangu-Weather 等主流气象大模型推理与性能评测（RMSE/ACC）。 |
| | 药物研发（推理） | NVIDIA | ⏳ 敬请期待 | 计划实现药物发现任务性能评测，包括分子对接、药物筛选、蛋白质折叠预测等生物医学计算应用。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 生命科学（推理） | NVIDIA | ⏳ 敬请期待 | 计划实现生命科学研究任务性能评测，包括基因组学分析、蛋白质结构预测、生物信息学计算等应用。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| 语音场景 | 语音识别（推理） | NVIDIA | [nvidia-audio-asr](skills/NVIDIA/audio/nvidia-audio-asr/SKILL.md) ✅ | 实现语音识别模型性能评测，包括吞吐量、字符错误率、词错误率等。 |
| | | Ascend | [ascend-audio-asr](skills/Ascend/audio/ascend-audio-asr/SKILL.md) ✅ | 支持SenseVoice等语音识别模型的推理和评测 |
| | 语音生成（推理） | NVIDIA | ⏳ 敬请期待 | 计划实现语音生成模型推理性能评测，包括文本到语音合成速度、语音质量等关键指标。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 音频理解（推理） | NVIDIA | ⏳ 敬请期待 | 计划实现音频理解任务性能评测，包括音频分类、声纹识别等。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 音频生成（推理） | NVIDIA | ⏳ 敬请期待 | 计划实现音频生成模型推理性能评测，包括音乐生成、音效合成等。 |
| | | 其他芯片 | ⏳ 敬请期待 | |

> 注：本列表会根据项目进展持续更新，欢迎贡献新的评测技能实现。

## 仓库结构

```
DeepEval-Skills/
├── skills/                          # Skill 目录（按芯片/场景嵌套组织）
│   ├── NVIDIA/                      # NVIDIA GPU 评测
│   │   ├── nlp/                     #   语言场景
│   │   │   ├── nvidia-nlp-training/      # NLP 训练评测
│   │   │   ├── nvidia-nlp-finetune/      # NLP 微调评测
│   │   │   ├── nvidia-nlp-inference/     # NLP 推理评测
│   │   │   └── nvidia-nlp-operator/      # NLP 算子评测
│   │   ├── cv/                      #   视觉场景
│   │   │   ├── nvidia-cv-detection/      # 目标检测训练评测
│   │   │   ├── nvidia-cv-pretrain/       # 图像分类训练评测
│   │   │   └── nvidia-cv-segmentation/   # 图像分割训练评测
│   │   ├── mm/                      #   多模态场景
│   │   │   ├── nvidia-mm-t2i/            # 文生图推理评测
│   │   │   └── nvidia-mm-t2v/            # 文生视频推理评测
│   │   ├── audio/                   #   语音场景
│   │   │   └── nvidia-audio-asr/         # 语音识别推理评测
│   │   └── render/                  #   渲染场景（规划中）
│   └── Hygon/                       # Hygon DCU 评测
│       └── science/                 #   科学计算场景
│           └── hygon-science-weather/    # 气象科学推理评测
├── template/                        # 新 Skill 模板
│   └── SKILL.md                     # 模板文件
└── README.md                        # 本文件
```

**命名约定**：
- Skill 目录路径按 `{芯片}/{场景}/{芯片}-{场景}-{任务类型}` 嵌套组织
  - 芯片：`NVIDIA`、`Hygon`、`Ascend` 等（首字母大写）
  - 场景：`nlp`、`cv`、`mm`（多模态）、`science`、`audio` 等
  - 任务类型：`training`、`finetune`、`inference`、`operator`、`detection`、`pretrain`、`segmentation`、`t2i`、`t2v`、`asr` 等
- Skill 目录名（末级）使用全小写 + 短横线连接，如 `nvidia-nlp-training`
- `SKILL.md` 的 `name` 字段必须与末级目录名一致

## 创建新评测 Skill

### 1. 创建目录并复制模板

```bash
# 按命名约定创建 skill 目录（嵌套结构）
mkdir -p skills/{芯片}/{场景}/{芯片}-{场景}-{任务类型}

# 复制模板文件
cp template/SKILL.md skills/{芯片}/{场景}/{芯片}-{场景}-{任务类型}/SKILL.md
```

### 2. 编辑 SKILL.md Frontmatter

修改 YAML frontmatter 中的元信息：

```yaml
---
name: {芯片}-{场景}-{任务类型}   # 必须与目录名一致
description: >
  描述该评测 Skill 的作用和触发条件。
  应清楚说明评测目标（芯片平台、模型、任务类型）和适用场景。
compatibility: "NVIDIA GPU / Hygon DCU / 其他芯片"
metadata:
  version: "1.0.0"
  category: training              # 评测任务类型: training, finetune, inference, operator
  scenario: nlp                   # 评测场景: nlp, cv, mm, science, audio
  tags: [benchmark, nvidia, nlp]  # 标签
---
```

### 3. 编写评测流程

完善 SKILL.md 正文，必须包含以下核心部分：

- **概述**：该评测 Skill 解决什么问题、何时触发、预期输入输出
- **硬件要求**：芯片类型与数量、显存/内存要求、存储要求
- **依赖要求**：Docker 镜像、预装框架与库
- **环境变量**：数据集路径、模型路径、输出目录等，以表格形式列出
- **执行流程**：逐步指南，包含完整的容器启动命令、数据准备脚本、评测执行命令
- **关键性能指标**：必采指标（如吞吐量、延迟、精度）和辅助指标（如 GPU 利用率、内存占用），以表格形式列出
- **指标采集命令**：提供可直接执行的 grep / Python 采集命令
- **常见问题**：典型的故障排查指南

可选内容：

- `scripts/` — 评测启动脚本、数据预处理脚本等
- `tools/` — 辅助工具（timer hook、数据转换等）
- `references/` — 参考文档

### 4. 验证

确保 Skill 目录结构正确：

```bash
# 检查 SKILL.md 是否存在且包含有效的 YAML frontmatter
head -10 skills/{芯片}/{场景}/{芯片}-{场景}-{任务类型}/SKILL.md

# 检查 name 字段与目录名是否一致
grep "^name:" skills/{芯片}/{场景}/{芯片}-{场景}-{任务类型}/SKILL.md
```

### 5. 更新 README

在本文档的「可用技能列表」表格中添加新 Skill 的条目。

> Skill 格式与内容规范详见 [Agent Skills 标准](https://agentskills.io/specification)。

## 许可证

暂无。
