# NVIDIA NLP 通用推理 Skill

本目录的 `nvidia-nlp-inference` 是基于 SGLang 的**通用文本推理评测 skill**。它提供统一的
容器输入输出、服务生命周期、离线压测、日志和 `result.json` 结果契约，适用于可由 SGLang
直接服务的 HuggingFace 文本生成模型。

具体的执行约束、容器变量和命令见 [SKILL.md](SKILL.md)。

## 目录说明

```text
nvidia-nlp-inference/
├── SKILL.md                       # 通用评测契约和流程
├── scripts/                       # 通用 serve / bench / calc 脚本
│   ├── common.sh                  # 解析模型/数据集路径，并校验正整数与 GPU 编号
│   ├── server.sh                  # 启动单机服务、等待就绪并记录 PID/日志
│   ├── serve_multi_host.sh        # 启动多机服务并检查各 rank 就绪
│   ├── bench.sh                   # 使用固定随机负载调用 sglang.bench_serving，写入压测日志
│   ├── calc.sh                    # 从 benchmark 日志提取指标，生成统一 result.json
│   └── <model>/                   # 仅该模型需要的定制脚本（可选）
└── references/
    ├── model_profiles.md          # 模型 profile 索引和通用配置
    ├── multi_host.md              # 多机执行流程
    └── models/<model>.md          # 模型专用参数、拓扑和执行顺序（可选）
```

## 在通用 skill 中接入模型

对于标准 SGLang 文本模型，先使用本目录的通用 `scripts/serve.sh`、`scripts/bench.sh` 和
`scripts/calc.sh`。模型的 TP、长度、超时、数据集优先级等参数写入
`references/models/<model>.md`，并在 `references/model_profiles.md` 登记模型键。

只有通用脚本无法满足模型的服务生命周期、压测方式或结果契约时，才在
`scripts/<model>/` 新增定制脚本；对应的输入变量、调用顺序和适用条件必须同时写入该模型的
reference。不要把模型专有逻辑写回通用脚本。

模型专用 skill 与此处的通用模型 profile 是不同的接入方式。若选择为某模型新建独立 skill，
应在其自身目录维护完整的 `SKILL.md`、脚本和 reference，而不是假定它会读取本目录的内容。
