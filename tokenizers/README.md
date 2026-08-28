# Tokenizer variants

这个目录把不同阶段的 tokenizer 实现隔离保存：

- `basic_bpe/`：2026-08-28 存档的基础 byte-level BPE。它只把 corpus item
  当作边界，因此可以学习 `dog.`、`dog?`、`dog ` 一类跨字符类别的 merge。
- `split_bpe/`：在 BPE 前增加可解释的 Unicode 预切分，并支持配置
  `protected characters`。训练和编码使用同一个 split policy。

两个目录都保留自己的核心代码、测试和本地前端，可分别运行，不共享运行时
状态。根目录原有入口继续保留，避免破坏已经在进行的学习记录与演示。
