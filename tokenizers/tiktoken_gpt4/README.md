# tiktoken-gpt-4

这是一个面向学习和可视化的轻量封装。它固定调用
`tiktoken.encoding_for_model("gpt-4")`，当前得到官方 `cl100k_base`
encoding；项目不复制 tiktoken 的 BPE 算法、merge ranks 或词表。

这个 tokenizer 不需要训练。它提供严格检查的 `encode`、`decode`、
`token_bytes`，以及能展示每个 token 的 UTF-8 byte offset、原始 bytes、hex
和安全文本表示的 `analyze`。

## 独立安装

需要 Python 3.10 或更新版本。在这个目录创建独立环境并安装已验证的固定版本：

```bash
cd tokenizers/tiktoken_gpt4
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

首次创建 tokenizer 时，tiktoken 可能需要联网下载 encoding 数据并写入本地
缓存；缓存命中后可重复离线使用。需要指定缓存位置时，可以设置
`TIKTOKEN_CACHE_DIR`。

## Python 使用

从仓库根目录使用上面创建的 `.venv` Python 运行：

```python
from tokenizers.tiktoken_gpt4 import GPT4Tokenizer

tokenizer = GPT4Tokenizer()
tokens = tokenizer.encode("你好，GPT-4 🙂\n")
text = tokenizer.decode(tokens)
report = tokenizer.analyze(text)

print(tokens)
print(report.token_details[0].as_dict())
```

普通字符串编码始终使用 `encode_ordinary`。因此字面文本
`<|endoftext|>` 会像普通用户文字一样被拆分，不会被解释成 special token，
也不会触发 tiktoken 对 special token 的检查错误。

这里报告的是传给 `encode` 的 **raw text token 数**。它不包含 ChatML、角色、
消息边界或 OpenAI API 在 raw text 之外可能添加的任何消息包装；不能直接把
它当作完整 chat request 的计费 token 数。

## JSON CLI

```bash
tokenizers/tiktoken_gpt4/.venv/bin/python \
  -m tokenizers.tiktoken_gpt4.tokenizer \
  --text 'hello 世界 🙂'
```

标准输出是 JSON，包含 tokenizer 元数据、token ids、round-trip 文本和逐 token
byte 明细。byte offset 是整个 UTF-8 byte stream 上的左闭右开区间。

## 本地前端

```bash
cd tokenizers/tiktoken_gpt4
./visualizations/run.sh 8012
```

浏览器访问 `http://127.0.0.1:8012/`。前端使用同一个 Python wrapper 生成
token ids 和 byte 明细，不在 JavaScript 中复制一份分词算法。

## 验证

安装依赖后，从仓库根目录运行：

```bash
tokenizers/tiktoken_gpt4/.venv/bin/python \
  -m unittest discover -s tokenizers/tiktoken_gpt4/tests -v

cd tokenizers/tiktoken_gpt4
.venv/bin/python -m unittest discover -s visualizations/tests -v
node --check visualizations/static/scripts/app.js
bash -n visualizations/run.sh
```

## 第三方依赖

本 variant 依赖 OpenAI 的 MIT-licensed
[`tiktoken`](https://github.com/openai/tiktoken)，当前固定并验证
`tiktoken==0.14.0`。仓库不复制它的源码、BPE ranks 或词表数据。
