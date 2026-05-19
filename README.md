# AIBenchAgent-skills

人工智能软硬件验证平台 评测skills

本仓库遵循 [Agent Skills 标准](https://agentskills.io/specification)。

## 简介
AIBenchAgent-skills 是一个专为人工智能软硬件验证平台设计的评测技能（Skills）集合。

该项目将常见的AI算子测试、AI模型测试等任务封装为可复用的技能单元，支持通过智能代理（Agent）动态调度执行，为AI芯片、加速卡、训练/推理框架以及端到端AI系统提供统一、可扩展的评测能力。


 
## 可用技能列表

下表列出当前可用的评测技能。


| **场景** | **子场景** | **芯片** | **Skills** | **说明** |
|---------|----------|--------|----------|---------|
| 语言场景 | 训练 | NVIDIA | [nvidia-nlp-training](skills/NVIDIA/nlp/nvidia-nlp-training) ✅ | 支持大规模语言模型预训练，包括模型初始化、数据加载、分布式训练、梯度同步等全流程性能评测。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 微调 | NVIDIA | [nvidia-nlp-finetune](skills/NVIDIA/nlp/nvidia-nlp-finetune)✅ | 计划实现语言模型指令微调、领域适配等任务的性能评测，涵盖LoRA、QLoRA等高效微调技术。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 推理 | NVIDIA | [nvidia-nlp-inference](skills/NVIDIA/nlp/nvidia-nlp-inference) ✅ | 计划实现语言模型在线推理性能评测，包括吞吐量、延迟、显存占用等关键指标。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 算子 | NVIDIA | [nvidia-nlp-operator](skills/NVIDIA/nlp/nvidia-nlp-operator) ✅ | 提供常见NLP基础算子（GEMM、Attention、FFN、LayerNorm等）性能测试，支持不同精度、批量大小、序列长度的组合测试。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| 视觉场景 | 检测训练 | NVIDIA | [nvidia-cv-detection](skills/NVIDIA/cv/nvidia-cv-detection) ✅ | 计划实现目标检测模型（如YOLO、Faster R-CNN）训练性能评测，包括数据增强、损失计算、后处理等环节。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 分类训练 | NVIDIA | [nvidia-cv-pretrain](skills/NVIDIA/cv/nvidia-cv-pretrain) ✅ | 计划实现图像分类模型（如ResNet、Vision Transformer）训练性能评测，涵盖图像预处理、模型前向/反向传播等。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 分割训练 | NVIDIA | [nvidia-cv-segmentation](skills/NVIDIA/cv/nvidia-cv-segmentation) ✅ | 计划实现图像分割模型（如U-Net、DeepLab）训练性能评测，包括像素级标注、分割掩码生成等任务。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 视觉感知推理 | NVIDIA | ⏳ 敬请期待 | 计划实现视觉感知任务性能评测，包括图像识别、物体检测、场景理解等综合视觉能力评估。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| 多模态场景 | 文生图推理 | NVIDIA | [nvidia-mm-t2i](skills/NVIDIA/mm/nvidia-mm-t2i) ✅ | 计划实现文本到图像生成模型（如Stable Diffusion、DALL-E）推理性能评测，包括提示词编码、扩散过程、图像解码等阶段。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 文生视频推理 | NVIDIA | [nvidia-mm-t2v](skills/NVIDIA/mm/nvidia-mm-t2v) ✅ | 计划实现文本到视频生成模型推理性能评测，涵盖时序建模、帧间一致性、视频质量评估等指标。 |
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
| 语音场景 | 语音识别（推理） | NVIDIA | [nvidia-audio-asr](skills/NVIDIA/audio/nvidia-audio-asr/SKILL.md) ✅ | 计划实现语音识别模型性能评测，包括吞吐量、字符错误率、词错误率等。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 语音生成（推理） | NVIDIA | ⏳ 敬请期待 | 计划实现语音生成模型推理性能评测，包括文本到语音合成速度、语音质量等关键指标。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 音频理解（推理） | NVIDIA | ⏳ 敬请期待 | 计划实现音频理解任务性能评测，包括音频分类、声纹识别等。 |
| | | 其他芯片 | ⏳ 敬请期待 | |
| | 音频生成（推理） | NVIDIA | ⏳ 敬请期待 | 计划实现音频生成模型推理性能评测，包括音乐生成、音效合成等。 |
| | | 其他芯片 | ⏳ 敬请期待 | |

> 注：本列表会根据项目进展持续更新，欢迎贡献新的技能实现。