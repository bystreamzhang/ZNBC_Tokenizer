# Split-aware BPE 本地前端

这个目录沿用基础版本的“静态浏览器界面 → Python adapter → 核心 tokenizer”
边界，但只保留新版需要的独立页面与 API：

```text
static/split.html + styles/split.css + scripts/split-app.js
                         │
                         │ POST /api/split-bpe/overview
                         ▼
adapters/split_overview.py
                         │
                         ▼
../bpe.py + ../pretokenizer.py
```

JavaScript 只渲染 Python 返回的数据，不重新实现 regex split、pair 统计、merge、
encode 或 decode。

启动：

```bash
./visualizations/run.sh
```

默认访问 `http://127.0.0.1:8010/`。server 仅使用 Python 标准库并只监听
`127.0.0.1`。

验证：

```bash
python3 -m unittest discover -s visualizations/tests -v
node --check visualizations/static/scripts/split-app.js
bash -n visualizations/run.sh
```
