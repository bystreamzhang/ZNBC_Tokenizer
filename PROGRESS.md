# PROGRESS

当前已提交的可运行版本位于 `tokenizers/basic_bpe/` 和 `tokenizers/split_bpe/`；下列 1～3 的基础 BPE、Encoder 与 Decoder 进度均可在 `basic_bpe` 存档中复现。

1. Byte pair encoding
    - 已实现 Unicode string -> UTF-8 byte token（`0~255`）
    - 已实现按频率逐次学习 `256, 257, ...` merge token
    - 已支持指定目标 `vocab_size`，以及对新 string 编码
    - 已支持输出 byte/token 长度、节省 token 数、压缩比和可选 token sequence
    - 已明确多个训练样本的边界和相同频率时的确定性规则
    - 已添加单元测试和命令行演示
    - 已将 `tokenizers/basic_bpe/bpe.py` 的注释和 docstring 改为更具体的中文说明
    - 已新增集中在 `tokenizers/basic_bpe/visualizations/` 的本地浏览器界面；页面通过 Python adapter 调用同 variant 的 `bpe.py`，核心算法不依赖展示代码
    - 已定义可复用的页面风格、view/adapter 目录约定和固定启动脚本

2. Byte-level BPE decoder
    - 已在 `BytePairEncoder` 上实现 `decode(int sequence) -> str`，使用当前训练词表把 token id 展开为 bytes
    - 已确保先拼接完整 byte stream 再做 UTF-8 解码，支持中文、emoji 和跨 token 的多字节字符
    - 已采用 `errors="replace"` 处理非法或不完整 UTF-8，输出 U+FFFD（�）；未知 token id 和非 int token 仍显式失败
    - 已添加 encode/decode round-trip、边界输入及错误输入测试
    - 已新增 Decoder 浏览器视图；它接收 Encoder 最近一次输出，展示 `ids → bytes → UTF-8 → Python string` 的过程
    - Decoder 页面调用 Python `BytePairEncoder.decode()`，浏览器不维护第二份解码算法

3. Byte-level BPE encoder（训练阶段与使用阶段分离）
    - 已明确 `encode(string) -> list[int]` 只重放训练得到的有序 merge rules，不会在待编码文本上重新统计 pair
    - 已补充中文注释，说明新 token 只可能触发后续规则、每条规则按顺序执行一次即可
    - 已增加三层级联合并、规则竞争、重叠 pair、未见 Unicode、空输入和错误类型等回归测试
    - 已新增独立 Encoder adapter 与浏览器视图，逐条展示固定规则的命中、跳过、合并前后序列和最终 int 列表
    - 可视化流程已拆分为 `BPE 构建 → Encoder → Decoder`，Encoder 输出会自动传给 Decoder

4. Split-aware byte-level BPE（避免不合理 merge）
    - 已将当前基础 `bpe.py`、测试和完整前端按 working tree 状态存档到 `tokenizers/basic_bpe/`，保留 `dog.` 等可跨类别 merge 的对照行为
    - 已新增隔离的 `tokenizers/split_bpe/`，训练与 encode 共用同一个 `RegexPretokenizer`
    - 已采用两阶段策略：先隔离 configured protected Unicode code points，再把其余文本划分为 letter / number / whitespace / other runs
    - 已实现强 protected-character barrier：不能与左右 merge，连续 protected 字符不能互相 merge，非 ASCII protected 字符自身的 UTF-8 bytes 也完全跳过 BPE
    - 已验证即使从其他汉字学到相同 UTF-8 byte prefix merge，protected `你` 仍保持原始 `[228, 189, 160]`
    - 已明确不包含 GPT-2 的 `'s`、`'re` 等英文缩写预设，不附着前导空格，也不做 Unicode normalization
    - 已新增独立本地前端，展示实际 protected/category regex、Unicode code points、ordered pieces、offset、bytes、piece 内 tokens、merge rules 和 round-trip 不变量
    - 已为基础存档、新 splitter、新 BPE 核心与新版 adapter/static routes 添加回归测试
    - 已将 split-aware 前端重构为 `构建词表 → Encode → Decode` 的紧凑单列流程；策略细节与大规模结果默认折叠，并完成桌面及窄屏响应式预览
    - 已完成 variant 迁移并删除根目录重复的 `bpe.py`、`tests/` 与 `visualizations/`；可运行入口统一位于 `tokenizers/basic_bpe/` 和 `tokenizers/split_bpe/`
