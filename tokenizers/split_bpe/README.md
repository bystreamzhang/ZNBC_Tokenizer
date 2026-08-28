# Split-aware byte-level BPE

这个版本保留基础实现“UTF-8 bytes → 按频率学习有序 merge rules → int token
sequence”的主线，在 BPE 前增加一层可解释、可配置的 pre-tokenization。

## 为什么先 split

基础 byte BPE 把整个 sample 当作一条 sequence，所以高频的 `dog.`、`dog?`、
`dog ` 可能各自占用不同的复合 token。GPT-2 的关键做法不是在 BPE 循环里
临时拉黑这些 pair，而是先把 string 切成 pieces，再分别执行 BPE。这样不同
piece 之间根本不会产生候选 pair。

本实现参考这个结构，但没有复制 GPT-2 的英文缩写列表，也没有“把一个前导
ASCII 空格附到后续单词”的例外。

## 当前 split policy

策略 id：`unicode-category-protected-v1`

1. 使用动态编译的 protected regex 找出用户配置的每个 Unicode code point。
2. 其余文本再按 `letter`、`number`、`whitespace`、`other` 分成连续 runs。
3. 普通 piece 可以在内部做 byte-level BPE，但不能跨 piece。
4. 每个 protected character 单独成为 `merge_allowed = false` 的 piece。
5. protected piece 直接输出原始 UTF-8 byte ids，完全跳过规则重放。
6. 所有 pieces 必须连续覆盖原 string，并满足拼接后与输入逐 code point 相同。

因此 protected character 有一个强保证：

- 不能与左侧 merge；
- 不能与右侧 merge；
- 连续出现时不能彼此 merge；
- 非 ASCII 字符自身的多个 UTF-8 bytes 也不能 merge；
- 即使词表从其他字符学到恰好匹配的 byte pair，encode 时仍不会应用到这个
  protected piece。

`protected_characters` 的单位是 Python Unicode code point，不是视觉上的
grapheme cluster；当前不执行 NFC/NFD normalization。比如 composed `é` 与
`e + ◌́` 是不同输入。Python 标准库 `re` 没有 `\p{L}`，所以基础 letter
规则使用 Unicode-aware `\w` 的零依赖近似，并用 fallback 保证不丢字符。

## Python 使用

从仓库根目录运行：

```python
from tokenizers.split_bpe import SplitAwareBytePairEncoder

encoder = SplitAwareBytePairEncoder(protected_characters=" .?!")
encoder.train(["dog.dog.", "dog?dog?", "dog dog "], vocab_size=272)

pieces = encoder.split("dog.dog?")
tokens = encoder.encode("dog.dog?")
text = encoder.decode(tokens)
```

命令行：

```bash
python3 -m tokenizers.split_bpe.bpe \
  --train-text 'dog.dog.' \
  --text 'dog.dog?' \
  --vocab-size 272 \
  --protected-chars ' .?!' \
  --show-tokens
```

## 本地前端

```bash
cd tokenizers/split_bpe
./visualizations/run.sh 8010
```

浏览器访问 `http://127.0.0.1:8010/`。页面展示：

- protected 与 category 两步实际 regex；
- protected code points、Unicode 编号与配置原文；
- 每个训练 sample 和待编码 string 的 ordered pieces；
- piece 类型、offset、UTF-8 bytes、merge 前后 token ids；
- 实际学习到的 merge rules；
- split 完整覆盖、protected 未变化、bytes 无损和 decode round-trip 不变量。

前端只渲染 `/api/split-bpe/overview` 返回的数据，不在 JavaScript 中维护第二份
split 或 BPE 算法。训练 corpus 输入框仍使用“一行一个 sample”、最多 200
个 samples 的约定；如需把换行作为 sample 内文本，请直接调用 Python/API，
或以后再扩展输入格式。

基础版本的完整存档与原前端位于 `../basic_bpe/`，建议分别使用 8008 与 8010
端口同时打开，对比跨边界 merge 的差异。

## 验证

```bash
cd tokenizers/split_bpe
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s visualizations/tests -v
bash -n visualizations/run.sh
./visualizations/run.sh --help
```

关键回归用例会先用 `佀佁佂佃佄` 学出 `(228, 189)`，再验证 protected `你`
仍编码为 `[228, 189, 160]`，用于捕获“训练时有边界、encode 却错误重放规则”
这一类隐蔽问题。

## 参考

- GPT-2 官方实现：<https://github.com/openai/gpt-2/blob/master/src/encoder.py>
- GPT-2 论文 `2.2 Input Representation：
  <https://cdn.openai.com/better-language-models/language-models.pdf>
