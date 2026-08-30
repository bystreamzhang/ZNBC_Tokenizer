# Own GPT-4 Tokenizer

这个目录按照 `exercise.md` 的 Step 1～4，从基础 byte-level BPE 开始，逐步实现
GPT-4 regex 预切分、`cl100k_base` merge 恢复、byte shuffle，以及真正的 special
token 处理。实际 `encode` / `decode` 都由本目录代码执行；`tiktoken` 只提供固定
词表数据并作为测试/页面中的 golden 对照。

## 三个 tokenizer

- `BasicTokenizer`：直接在完整 UTF-8 byte stream 上训练 BPE。
- `RegexTokenizer`：先使用 exercise 指定的 GPT-4 pattern 无损分块，再在每个
  piece 内独立训练或编码，提供 `train()` 和 special-token 注册能力。
- `GPT4Tokenizer`：恢复并固定使用 `cl100k_base` 的 100,000 条 merge rules，
  自行处理 byte permutation；它不能重训，训练自定义词表应使用
  `RegexTokenizer.train()`。

三个类都提供 `encode()`、`decode()` 和可观察的 `merges` / `vocab`。可训练类的
接口与练习保持一致：

```python
train(text, vocab_size, verbose=False)
encode(text)
decode(ids)
```

## 安装

需要 Python 3.10 或更新版本。依赖固定在本 variant 内：

```bash
cd tokenizers/own_gpt4
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd ../..
```

`tiktoken==0.14.0` 首次加载 `cl100k_base` 时可能需要联网获取 encoding 数据并
写入缓存；缓存命中后可以重复离线使用。

## 训练自己的 tokenizer

从仓库根目录运行：

```python
from tokenizers.own_gpt4 import BasicTokenizer, RegexTokenizer

basic = BasicTokenizer()
basic.train("你好你好 hello hello", vocab_size=272)
basic_ids = basic.encode("你好 hello")
assert basic.decode(basic_ids) == "你好 hello"

regex_tokenizer = RegexTokenizer()
regex_tokenizer.train("dog.dog 12341234 你好你好", vocab_size=280)
regex_ids = regex_tokenizer.encode("dog.dog 1234 你好")
assert regex_tokenizer.decode(regex_ids) == "dog.dog 1234 你好"
```

训练和 encode 使用同一个 regex split policy。数字最多三个一组，字母、数字、
标点和空白 piece 之间不会学习或执行跨边界 merge。每次重新调用 `train()` 都会
清空旧规则；同频 pair 使用 token-id pair 字典序决定胜者，结果可复现。

## 固定 GPT-4 / cl100k_base

```python
from tokenizers.own_gpt4 import GPT4Tokenizer

tokenizer = GPT4Tokenizer()
text = "hello world!!!? (안녕하세요!) lol123 😉"
ids = tokenizer.encode(text)

assert ids == [
    15339, 1917, 12340, 30, 320, 31495, 230,
    75265, 243, 92245, 16715, 28509, 4513, 57037,
]
assert tokenizer.decode(ids) == text
```

初始化时会从 tiktoken 的 `_mergeable_ranks` 恢复 parent merges，再从最初的 256
个 ranks 构造 byte shuffle。之后的 encode/decode 不调用 tiktoken 的编码器。
因为 `_mergeable_ranks` 是私有数据接口，本目录固定并测试 `tiktoken==0.14.0`，
升级依赖时必须重新跑完整 parity 测试。

## Special tokens

`GPT4Tokenizer` 默认注册五个 cl100k 常用项：

| 字面量 | Token ID |
|---|---:|
| `<|endoftext|>` | 100257 |
| `<|fim_prefix|>` | 100258 |
| `<|fim_middle|>` | 100259 |
| `<|fim_suffix|>` | 100260 |
| `<|endofprompt|>` | 100276 |

行为与 tiktoken 的安全约定一致：

```python
# 默认：遇到已注册 special literal 会抛 ValueError。
tokenizer.encode("<|endoftext|>hello")

# 允许并识别全部 special tokens。
ids = tokenizer.encode(
    "<|endoftext|>hello world",
    allowed_special="all",
)
assert ids == [100257, 15339, 1917]

# 只允许显式集合中的 special token。
tokenizer.encode(
    "<|fim_prefix|>code",
    allowed_special={"<|fim_prefix|>"},
)

# 关闭检查，把所有 special-looking 字面量当普通文本。
ordinary_ids = tokenizer.encode(
    "<|endoftext|>",
    disallowed_special=(),
)
assert 100257 not in ordinary_ids
```

为避免误把一个裸字符串当成“字符集合”，本实现要求显式
`disallowed_special` 使用字符串 collection（例如 set/tuple），不接受除
`"all"` 之外的裸 `str`。这是比 tiktoken 更严格的输入校验，不影响上面这些
正常调用方式。

`RegexTokenizer` 可通过 `register_special_tokens({literal: id})` 注册自定义映射；
special id 不能与普通词表或其他 special id 冲突。

## JSON CLI

固定 GPT-4：

```bash
tokenizers/own_gpt4/.venv/bin/python \
  -m tokenizers.own_gpt4.cli \
  --text '<|endoftext|>hello world' \
  --special-policy all
```

训练模式：

```bash
tokenizers/own_gpt4/.venv/bin/python \
  -m tokenizers.own_gpt4.cli \
  --mode train \
  --train-text 'abababab 你好你好' \
  --vocab-size 272 \
  --text 'abab 你好'
```

## 自动测试

安装依赖后，从仓库根目录运行：

```bash
tokenizers/own_gpt4/.venv/bin/python \
  -m unittest discover -s tokenizers/own_gpt4/tests -v

tokenizers/own_gpt4/.venv/bin/python \
  -m unittest discover -s tokenizers/own_gpt4/visualizations/tests -v

node --check tokenizers/own_gpt4/visualizations/static/scripts/app.js
bash -n tokenizers/own_gpt4/visualizations/run.sh
tokenizers/own_gpt4/visualizations/run.sh --help
```

核心测试包含 exercise golden、多语言/emoji、special policies、invalid/reserved
ids、byte shuffle 双射，以及 100,256 个 mergeable token bytes 的全量 tiktoken
对照。测试还会 mock tiktoken 的 `encode` / `decode`，防止自研实现退化成 wrapper。

## 本地前端

```bash
cd tokenizers/own_gpt4
./visualizations/run.sh 8014
```

浏览器访问 `http://127.0.0.1:8014/`。页面提供两个模式：

1. **固定 GPT-4 / cl100k_base**：展示自研 ids、逐 token bytes、special token
   类型和 tiktoken golden 一致性。
2. **训练自己的 Regex BPE**：输入训练 string 和目标词表大小，查看 regex pieces、
   学到的 merge rules、encode 与 decode round-trip。

建议手工检查：

- exercise Step 3 的韩文/emoji 文本与 tiktoken ids 完全一致；
- `<|endoftext|>hello world` 在“允许全部”时得到
  `[100257, 15339, 1917]`；
- 同一输入在“默认拒绝”时报明确错误；
- 在“全部作为普通文本”时不产生 `100257`，但仍能 round-trip；
- 训练模式中待编码文本不参与训练，修改训练 string 后 merge rules 才变化。

前端 JavaScript 只渲染 Python adapter 返回的数据，不包含第二份 tokenizer
算法。server 只监听 `127.0.0.1`，并限制请求大小、静态文件目录和安全响应头。

更多前端结构说明见 [visualizations/README.md](visualizations/README.md)。

## 实现边界

- 保留输入的原始 Unicode 序列，不执行 NFC/NFD normalization。
- decode 先拼接完整 bytes，再使用 `errors="replace"` 解码非法 UTF-8。
- GPT-4 regex 固定使用 `exercise.md` 给出的 pattern，不在运行时读取 tiktoken
  的私有 `_pat_str`。
- `cl100k_base` nominal vocabulary 含 reserved gaps；不能只按
  `0 <= token_id < vocab_size` 判断 id 有效性。
- SentencePiece / Unicode code-point BPE（exercise Step 5）不属于本 variant。
