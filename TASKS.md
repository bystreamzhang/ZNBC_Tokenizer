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
- [x] decode：从 token sequence 恢复原 string
- [ ] 保存和加载训练好的 vocabulary / merge rules
- [x] 基于当前 Python BPE 实现的本地浏览器可视化界面
- [ ] 后续 tokenizer 功能沿用各 variant 内 `visualizations/` 的 adapter/view/style 约定增加展示
- [x] 将 BPE 构建与 Encoder 使用阶段拆开，并展示有序 merge 的级联过程

分词器边界优化：

- [x] 存档当前基础 byte-level BPE 与对应可运行前端
- [x] 在独立目录实现训练/编码共用的 regex pre-tokenization
- [x] 支持配置永远不参与 merge 的 protected Unicode characters
- [x] 前端展示当前 split policy 与每个实际 split piece
- [x] 覆盖 ASCII/Unicode/连续分隔符/换行/正则元字符与对抗 merge 测试
- [x] 删除迁移后的根目录重复实现，每个 tokenizer variant 独立维护核心、测试与 `visualizations/`
- [ ] 后续考虑以 longest-match 支持多 code point protected literals
- [ ] 后续评估是否用第三方 `regex` 的 Unicode properties 替换标准库 `re` 近似

tiktoken GPT-4 对照实现：

- [x] 在独立 variant 中直接调用 tiktoken 的 GPT-4 / `cl100k_base` tokenizer
- [x] 提供 Python API、JSON CLI、独立 README、依赖与使用方法
- [x] 提供风格一致的本地前端，展示 token ids、bytes 与 decode round-trip
- [x] 覆盖普通文本、Unicode、emoji、special-token 字面量、无效 id 与前端安全回归测试
