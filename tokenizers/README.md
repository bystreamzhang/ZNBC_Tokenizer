# Tokenizer variants

这个目录把不同阶段的 tokenizer 实现隔离保存：

- `basic_bpe/`：2026-08-28 存档的基础 byte-level BPE。它只把 corpus item
  当作边界，因此可以学习 `dog.`、`dog?`、`dog ` 一类跨字符类别的 merge。
- `split_bpe/`：在 BPE 前增加可解释的 Unicode 预切分，并支持配置
  `protected characters`。训练和编码使用同一个 split policy。
- `tiktoken_gpt4/`：直接调用 `tiktoken.encoding_for_model("gpt-4")` 的
  `cl100k_base` 固定词表，提供轻量 Python API、JSON CLI 与逐 token bytes
  前端；不在本仓库重新实现 GPT-4 的 BPE。
- `own_gpt4/`：按照练习自行实现 `BasicTokenizer`、可训练的 GPT-4 regex
  `RegexTokenizer`，并从 `cl100k_base` ranks 恢复 merges 与 byte shuffle；
  支持常见 special tokens、JSON CLI、独立测试和双模式本地前端。

四个目录都保留自己的核心代码、测试和本地前端，可分别运行，不共享运行时
状态。原根目录的基础实现已完整迁移到 `basic_bpe/`，不再保留重复运行入口。
