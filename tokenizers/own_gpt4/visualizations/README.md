# Own GPT-4 可视化

这个前端沿用仓库已有 variant 的隔离约定：静态页面只负责收集输入和渲染 JSON，
训练、regex split、merge、special-token 识别、encode、token bytes 与 decode 全部
调用 `tokenizers/own_gpt4/` 的 Python 核心。

## 目录职责

- `adapters/own_gpt4_overview.py`：校验输入，调用真实 tokenizer，生成稳定 JSON。
- `server.py`：标准库 HTTP server、静态文件和单一 overview API。
- `static/`：HTML/CSS/JavaScript 展示层，不实现 BPE。
- `tests/test_visualizations.py`：adapter、DOM、server、安全头及响应式约束测试。
- `run.sh`：定位独立虚拟环境并只监听 `127.0.0.1`。

API：

```text
GET  /api/health
POST /api/own-gpt4/overview
```

POST payload 分为两种：

```json
{
  "mode": "gpt4",
  "special_policy": "all",
  "text": "<|endoftext|>hello world"
}
```

```json
{
  "mode": "train",
  "special_policy": "none_raise",
  "training_text": "abababab 你好你好",
  "vocab_size": 272,
  "text": "abab 你好"
}
```

待编码文本最多 20,000 UTF-8 bytes；训练文本最多 5,000 UTF-8 bytes，页面训练
词表限制为 256～512。这个限制只保护交互页面，Python `train()` API 不限制目标
词表大小。
固定词表只返回前 160 条 merge 预览，完整 merge 总数仍单独报告；浏览器也对
pieces、merges 和 token cards 设置展示上限，但完整 encoded ids 始终保留。

## 启动与测试

从仓库根目录安装依赖后：

```bash
cd tokenizers/own_gpt4
./visualizations/run.sh 8014
```

访问 `http://127.0.0.1:8014/`。自动测试命令见上级 [README](../README.md)。
