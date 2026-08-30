# tiktoken GPT-4 本地前端

这个目录提供一个独立的本地页面，用核心 `GPT4Tokenizer` 展示固定的
`gpt-4 → cl100k_base` 映射、encode 结果和 decode round-trip：

```text
static/index.html + styles/ + scripts/app.js
                         │
                         │ POST /api/tiktoken-gpt-4/overview
                         ▼
adapters/tiktoken_overview.py
                         │
                         ▼
../tokenizer.py → tiktoken
```

JavaScript 只渲染 Python adapter 返回的数据，不重新实现 tokenization。页面展示
逐 token 的 id、byte offsets、原始 bytes、hex 和安全 UTF-8 文本；token ids 始终
完整保留，逐 token 卡片最多显示前 300 项。

## 启动

可以先从仓库根目录创建 variant 自己的虚拟环境并安装依赖：

```bash
python3 -m venv tokenizers/tiktoken_gpt4/.venv
tokenizers/tiktoken_gpt4/.venv/bin/python -m pip install \
  -r tokenizers/tiktoken_gpt4/requirements.txt
```

然后启动仅监听本机的 server：

```bash
./tokenizers/tiktoken_gpt4/visualizations/run.sh
```

默认访问 `http://127.0.0.1:8012/`；也可以把端口作为唯一参数传入，例如
`./tokenizers/tiktoken_gpt4/visualizations/run.sh 9000`。脚本会优先使用
`tokenizers/tiktoken_gpt4/.venv/bin/python`，不存在时才使用 PATH 中的
`python3`；依赖检查与 server 始终使用同一个 Python。

这里只对输入的 raw text 使用 `cl100k_base`。它不会估算 ChatML 或 OpenAI API
消息包装产生的额外 tokens，不能直接当作完整请求的计费 token 数。

## 验证

```bash
cd tokenizers/tiktoken_gpt4
.venv/bin/python -m unittest discover -s visualizations/tests -v
node --check visualizations/static/scripts/app.js
bash -n visualizations/run.sh
```
