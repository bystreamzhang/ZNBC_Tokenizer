# Basic byte-level BPE（存档）

这是原根目录基础分词器及其前端在 2026-08-28 的 working-tree 迁移快照。存档
包含当时尚未提交的 Encoder / Decoder 页面，不能用旧 commit 重新生成。
核心与静态页面内容保持快照行为；仅把 Python import 和启动入口改为包相对
路径，确保它不会因当前 working directory 不同而误用其他 tokenizer 实现。

这个版本只把不同 corpus item 当作边界；一个 item 内的整段 string 会先变成
同一条 UTF-8 byte sequence，再执行 BPE。因此它被保留下来作为对照组，用于
观察 `dog.`、`dog?`、`dog ` 等跨类别 merge 为什么会出现。

运行测试：

```bash
cd tokenizers/basic_bpe
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s visualizations/tests -v
```

启动存档前端：

```bash
cd tokenizers/basic_bpe
./visualizations/run.sh 8008
```

页面默认只监听 `127.0.0.1`。这个目录是完整快照，运行时会 import 同目录的
`bpe.py`，不会调用新版 split-aware tokenizer。
