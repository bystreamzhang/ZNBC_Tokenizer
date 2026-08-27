# README.md

参考教程：

- https://www.youtube.com/watch?v=zduSFxRajkE
- https://cs336.stanford.edu/

说分词器很重要，所以学着做一下。

- @PROMPTS : 记录我各个工作对话的PROMPTS，因为我自己不写代码，PROMPTS其实就是我的“代码”，包含了最凝练的规范要求等，也可能存在某些问题可以溯源。并且某些情况下可以预写一些prompts而不是等待agents跑完才写。

## Byte Pair Encoding（第一步）

`bpe.py` 是一个只依赖 Python 标准库的 byte-level BPE 实现：

1. Unicode string 先编码成 UTF-8 bytes，初始 token id 是 `0~255`。
2. 训练时反复统计相邻 pair，合并频率最高的一对，并依次分配 `256`、`257`……。
3. `vocab_size` 包含最初的 256 个 byte token，因此最多训练
   `vocab_size - 256` 次 merge。
4. 编码新文本时按照训练得到的顺序重放 merge，不会在新文本上重新统计。

最小 Python 用法：

```python
from bpe import BytePairEncoder

tokenizer = BytePairEncoder()
training = tokenizer.train(
    ["你好你好", "hello hello"],
    vocab_size=264,
)

tokens = tokenizer.encode("你好 hello")
summary = tokenizer.analyze("你好 hello", include_tokens=True)

print(training.as_dict())
print(tokens)
print(summary.as_dict())
```

也可以直接从命令行观察结果：

```bash
python3 bpe.py \
  --train-text '你好你好你好' \
  --train-text 'hello hello hello' \
  --text '你好 hello' \
  --vocab-size 264 \
  --show-tokens
```

其中 `compression_ratio = UTF-8 byte 数 / BPE token 数`；例如 `4.0`
表示平均每个 BPE token 承载 4 个原始 byte token。`reduction_ratio` 则表示
token 数量缩减的比例。

多个训练 string 保留各自边界，最后一个 string 的尾 byte 不会和下一个
string 的首 byte 组成 pair。频率相同时选择数值字典序最小的 pair，以保证
结果可复现。

这一版实现的是纯 byte-level BPE，还没有加入 GPT 系列 tokenizer 常见的
regex/pre-tokenization。因此，同一个训练 string 内的 pair 可以跨 Unicode
字符或空格边界。训练也采用每轮重新扫描语料的直观写法，适合学习和正确性
检查，暂时不是面向大语料的高性能实现。

运行测试：

```bash
python3 -m unittest discover -s tests -v
```
