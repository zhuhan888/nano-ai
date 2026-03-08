# 纳米智能

纳米智能是一个使用 `Python + numpy` 实现的超小型本地对话模型，目标是在资源受限环境中完成基础单轮问答。

## 特性

- 纯 `numpy` 的 Decoder-only Transformer
- 字符级 tokenizer，默认中文友好
- 合成数据生成脚本，默认约 10000 条独立问答样本
- 训练进度显示、checkpoint 保存、断点续训
- 训练样本编码缓存，重复训练/续训时减少数据准备时间
- 命令行聊天，支持流式输出
- **无历史记录**：每次输入都是独立会话，不记住上一轮内容

## 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 使用方式

### 1. 生成训练数据

```bash
python data_generator.py
```

生成文件：`data/generated.jsonl`

### 2. 开始训练

默认会自动拉起纯 Python 白色训练 CLI：

```bash
python train.py
```

断点续训：

```bash
python train.py --resume
```

默认会输出：

- 当前训练步数和进度百分比
- loss
- 学习率
- checkpoint 保存日志
- 是否命中 `data/encoded_dataset.npz` 编码缓存

### 3. 启动聊天

默认会自动拉起纯 Python 白色聊天 CLI：

```bash
python chat.py
```

聊天说明：

- 每次输入都独立处理
- 不支持记忆“上一句说过什么”
- 输入 `quit` 或 `exit` 退出

## 项目结构

```text
nano-intelligence/
├── README.md
├── config.py
├── model.py
├── tokenizer.py
├── data_generator.py
├── data_crawler.py
├── train.py
├── chat.py
├── requirements.txt
├── data/
└── checkpoints/
```

## 性能说明

- 首版优先保证“能训练、能恢复、能对话”
- 需求文档中的理想性能指标作为优化方向，不作为首次交付硬门槛
- 在移动设备/Termux 环境中，可通过减小 `batch_size`、减少 epoch 数降低内存压力

## 常见问题

### 为什么它记不住上一轮？

这是当前版本的设计选择。训练数据、推理逻辑、命令行会话都按单轮独立问答实现。

### 为什么回答还不够聪明？

这是一个超小模型，并且首版只使用合成数据，能力会明显弱于大模型。
