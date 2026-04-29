# AIBenchAgent-skills

人工智能软硬件验证平台 评测skills

本仓库遵循 [Agent Skills 标准](https://agentskills.io/specification)。

## 简介
AIBenchAgent-skills 是一个专为人工智能软硬件验证平台设计的评测技能（Skills）集合。

该项目将常见的AI算子测试、AI模型测试等任务封装为可复用的技能单元，支持通过智能代理（Agent）动态调度执行，为AI芯片、加速卡、训练/推理框架以及端到端AI系统提供统一、可扩展的评测能力。



## 可用技能列表

下表列出当前可用的评测技能。


| 场景 | 子场景 | 芯片 | Skills | 说明 |
|------|--------|------|------|------|
<<<<<<< Updated upstream
| 语言场景 | 训练 | NVIDIA_H200 | [h200-nlp-training](skills/NVIDIA_H200/nlp/h200-nlp-training) ✅ | 支持大规模语言模型预训练，包括模型初始化、数据加载、分布式训练、梯度同步等全流程性能评测。 |
| 语言场景 | 微调 | NVIDIA_H200 | ⏳ 敬请期待 | 计划实现语言模型指令微调、领域适配等任务的性能评测，涵盖LoRA、QLoRA等高效微调技术。 |
| 语言场景 | 推理 | NVIDIA_H200 | [h200-nlp-inference](skills/NVIDIA_H200/nlp/h200-nlp-inference) ✅ | 计划实现语言模型在线推理性能评测，包括吞吐量、延迟、显存占用等关键指标。 |
| 语言场景 | 算子 | NVIDIA_H200 | [h200-nlp-operator](skills/NVIDIA_H200/nlp/h200-nlp-operator) ✅ | 提供常见NLP基础算子（GEMM、Attention、FFN、LayerNorm等）性能测试，支持不同精度、批量大小、序列长度的组合测试。 |
| 视觉场景 | 检测训练 | NVIDIA_H200 | [h200-cv-detection](skills/NVIDIA_H200/cv/h200-cv-detection) ✅ | 计划实现目标检测模型（如YOLO、Faster R-CNN）训练性能评测，包括数据增强、损失计算、后处理等环节。 |
| 视觉场景 | 分类训练 | NVIDIA_H200 | [h200-cv-pretrain](skills/NVIDIA_H200/cv/h200-cv-pretrain) ✅ | 计划实现图像分类模型（如ResNet、Vision Transformer）训练性能评测，涵盖图像预处理、模型前向/反向传播等。 |
| 视觉场景 | 分割训练 | NVIDIA_H200 | [h200-cv-segmentation](skills/NVIDIA_H200/cv/h200-cv-segmentation) ✅ | 计划实现图像分割模型（如U-Net、DeepLab）训练性能评测，包括像素级标注、分割掩码生成等任务。 |
| 多模态场景 | 文生图推理 | NVIDIA_H200 | ⏳ 敬请期待 | 计划实现文本到图像生成模型（如Stable Diffusion、DALL-E）推理性能评测，包括提示词编码、扩散过程、图像解码等阶段。 |
| 多模态场景 | 文生视频推理 | NVIDIA_H200 | ⏳ 敬请期待 | 计划实现文本到视频生成模型推理性能评测，涵盖时序建模、帧间一致性、视频质量评估等指标。 |
| 渲染仿真场景 | 仿真 | NVIDIA_H200 | ⏳ 敬请期待 | 计划实现渲染仿真任务性能评测，包括光线追踪、物理仿真、实时渲染等场景。 |
=======
| 语言场景 | 训练 | NVIDIA | [nvidia-nlp-training](skills/NVIDIA/nlp/nvidia-nlp-training) ✅ | 支持大规模语言模型预训练，包括模型初始化、数据加载、分布式训练、梯度同步等全流程性能评测。 |
| 语言场景 | 微调 | NVIDIA | [nvidia-nlp-finetune](skills/NVIDIA/nlp/nvidia-nlp-finetune)✅ | 计划实现语言模型指令微调、领域适配等任务的性能评测，涵盖LoRA、QLoRA等高效微调技术。 |
| 语言场景 | 推理 | NVIDIA | [nvidia-nlp-inference](skills/NVIDIA/nlp/nvidia-nlp-inference) ✅ | 计划实现语言模型在线推理性能评测，包括吞吐量、延迟、显存占用等关键指标。 |
| 语言场景 | 算子 | NVIDIA | [nvidia-nlp-operator](skills/NVIDIA/nlp/nvidia-nlp-operator) ✅ | 提供常见NLP基础算子（GEMM、Attention、FFN、LayerNorm等）性能测试，支持不同精度、批量大小、序列长度的组合测试。 |
| 视觉场景 | 检测训练 | NVIDIA | [nvidia-cv-detection](skills/NVIDIA/cv/nvidia-cv-detection) ✅ | 计划实现目标检测模型（如YOLO、Faster R-CNN）训练性能评测，包括数据增强、损失计算、后处理等环节。 |
| 视觉场景 | 分类训练 | NVIDIA | [nvidia-cv-pretrain](skills/NVIDIA/cv/nvidia-cv-pretrain) ✅ | 计划实现图像分类模型（如ResNet、Vision Transformer）训练性能评测，涵盖图像预处理、模型前向/反向传播等。 |
| 视觉场景 | 分割训练 | NVIDIA | [nvidia-cv-segmentation](skills/NVIDIA/cv/nvidia-cv-segmentation) ✅ | 计划实现图像分割模型（如U-Net、DeepLab）训练性能评测，包括像素级标注、分割掩码生成等任务。 |
| 多模态场景 | 文生图推理 | NVIDIA | ⏳ 敬请期待 | 计划实现文本到图像生成模型（如Stable Diffusion、DALL-E）推理性能评测，包括提示词编码、扩散过程、图像解码等阶段。 |
| 多模态场景 | 文生视频推理 | NVIDIA | ⏳ 敬请期待 | 计划实现文本到视频生成模型推理性能评测，涵盖时序建模、帧间一致性、视频质量评估等指标。 |
| 渲染仿真场景 | 仿真 | NVIDIA | ⏳ 敬请期待 | 计划实现渲染仿真任务性能评测，包括光线追踪、物理仿真、实时渲染等场景。 |
>>>>>>> Stashed changes
| （待补充） | （待补充） | （其他芯片） | ⏳ 敬请期待 | 更多芯片支持规划中，包括但不限于国产AI芯片、其他品牌GPU等。技能将根据不同芯片架构进行适配优化。 |

> 注：本列表会根据项目进展持续更新，欢迎贡献新的技能实现。