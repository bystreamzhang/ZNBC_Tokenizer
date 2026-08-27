# Tokenizer 可视化

这个目录集中保存本地浏览器展示所需的 server、adapter、静态界面、统一
样式、测试和说明。核心算法仍在仓库根目录的 `bpe.py`，不 import 本目录的
任何代码。

## 架构边界

```text
浏览器中的静态界面
    │  POST /api/bpe/trace（只提交文本和 vocab_size）
    ▼
visualizations/server.py
    │
    ▼
visualizations/adapters/bpe_trace.py
    │  调用 BytePairEncoder.train() / encode() / analyze()
    │  使用实际 merges 重放并生成逐轮 trace
    ▼
bpe.py
```

浏览器 JavaScript 不实现 pair 统计、pair 选择或 merge 算法，只渲染 Python
返回的 JSON。因此修改 `bpe.py` 后，重新点击页面上的“用 bpe.py 运行”即可
观察当前实现的结果，不需要同步维护第二份 BPE 算法。

## 启动

在仓库根目录执行：

```bash
./visualizations/run.sh
```

然后在浏览器访问：

```text
http://127.0.0.1:8008/
```

使用其他端口：

```bash
./visualizations/run.sh 9000
```

按 `Ctrl+C` 停止。脚本默认只绑定 `127.0.0.1`，不会把服务公开到局域网或
公网。

如果代码位于 Remote-SSH 机器，在 VS Code 的 **Ports** 面板转发 `8008`
端口，然后打开 VS Code 给出的 forwarded address。无需在远程机器上绑定
`0.0.0.0`。

## 当前页面能检查什么

- 每一轮真实 BPE merge 选择的 pair、频率和新 token id。
- merge 前后的多个独立 token sequence。
- 重叠 pair 的统计频率与实际非重叠合并次数。
- 训练语料的 token 数和压缩比。
- 使用训练结果编码另一个 string 的真实 token sequence。
- trace 是否与 `BytePairEncoder.encode()` 一致。
- token 对应的 bytes 是否能够还原待编码 string。

## 目录约定与统一风格

```text
visualizations/
├── adapters/              # 核心算法 -> 展示 JSON；不包含页面代码
├── static/
│   ├── index.html         # 页面结构和视图入口
│   ├── scripts/
│   │   ├── api.js         # HTTP 调用
│   │   ├── ui.js          # 可复用展示组件
│   │   └── views/         # 各算法自己的页面 controller
│   └── styles/
│       ├── system.css     # 统一颜色、字体、间距和通用组件
│       └── bpe.css        # 只属于 BPE view 的样式
├── tests/                 # 展示 adapter 和 server 的测试
├── server.py              # 静态文件和只读 JSON API
└── run.sh                 # 固定启动入口
```

以后添加其他 tokenizer 展示时：

1. 在 `adapters/` 添加只调用核心代码的 adapter。
2. 在 `static/scripts/views/` 添加对应 view，不在浏览器复制核心算法。
3. 优先复用 `system.css` 的 design tokens 和通用组件。
4. 只把该视图特有的样式放进新的 view stylesheet。
5. 在 `tests/` 验证 adapter 输出确实来自核心实现。

## 验证

这些命令不会启动常驻 server：

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s visualizations/tests -v
bash -n visualizations/run.sh
./visualizations/run.sh --help
```

完整的浏览器运行需要由你执行 `./visualizations/run.sh` 后验证；离线测试只
覆盖 adapter、静态文件路径和核心数据不变量。

