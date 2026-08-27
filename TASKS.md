# TASKS

目标：

1. 输入strings，喂到LLM
    1. 为此，需要把string做分词(tokenize)，变成一些int（以一个固定的词汇表, vocabulary）
    2. 我们会用这些int，在一个lookup table of vectors里找vectors，然后把这些vectors喂给LLM作为input
2. 我们不想只支持English，还想支持其他语言甚至emoji

当前实现任务：

- [x] byte-level Byte Pair Encoding 训练
- [x] 使用训练得到的 merge 编码新 string
- [x] 输出长度、token sequence（可选）和压缩信息
- [ ] decode：从 token sequence 恢复原 string
- [ ] 保存和加载训练好的 vocabulary / merge rules
- [x] 基于当前 Python BPE 实现的本地浏览器可视化界面
- [ ] 后续 tokenizer 功能沿用 `visualizations/` 的 adapter/view/style 约定增加展示
