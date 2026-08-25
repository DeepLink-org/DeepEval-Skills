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
│   ├── serve.sh
│   ├── bench.sh
│   ├── calc.sh
│   └── <model>/                   # 仅该模型需要的定制脚本（可选）
└── references/
    ├── model_profiles.md          # 模型 profile 索引和通用配置
    └── models/<model>.md          # 模型专用参数、拓扑和执行顺序（可选）
```

## 通用 skill 与模型专用 skill 的关系

`nvidia-nlp-inference` 与 `deepseek_r1/`、`llama2_7b/` 等每模型一个目录的专用 skill
是**并列且相互独立**的 skill：

- 通用 skill 不会自动调用、继承或修改任何模型专用 skill 的脚本、reference 或结果；
- 模型专用 skill 也不会自动复用本目录的通用脚本；其行为仅由该目录自身的 `SKILL.md` 定义；
- 一次评测应根据任务选择其中一个 skill 的流程，不应同时串联两套流程；
- 它们可以使用相同的镜像、模型或数据集，但这不表示存在代码或配置依赖。

## 在通用 skill 中接入模型

对于标准 SGLang 文本模型，先使用本目录的通用 `scripts/serve.sh`、`scripts/bench.sh` 和
`scripts/calc.sh`。模型的 TP、长度、超时、数据集优先级等参数写入
`references/models/<model>.md`，并在 `references/model_profiles.md` 登记模型键。

只有通用脚本无法满足模型的服务生命周期、压测方式或结果契约时，才在
`scripts/<model>/` 新增定制脚本；对应的输入变量、调用顺序和适用条件必须同时写入该模型的
reference。不要把模型专有逻辑写回通用脚本。

模型专用 skill 与此处的通用模型 profile 是不同的接入方式。若选择为某模型新建独立 skill，
应在其自身目录维护完整的 `SKILL.md`、脚本和 reference，而不是假定它会读取本目录的内容。
