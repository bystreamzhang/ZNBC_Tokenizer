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

5. tiktoken GPT-4 tokenizer（固定官方词表对照）
    - 已新增独立的 `tokenizers/tiktoken_gpt4/`，固定通过 `tiktoken.encoding_for_model("gpt-4")` 使用 `cl100k_base`，不复制或重新实现 BPE、merge ranks 与词表
    - 已提供 `GPT4Tokenizer` 的 `encode`、`decode`、`token_bytes`、`analyze` 与 JSON CLI；普通输入使用 `encode_ordinary`，special-token 字面量不会被当作控制 token
    - 已展示每个 token 的 id、UTF-8 byte offset、原始 bytes、hex 与安全文本，并验证 bytes 重建和 Unicode / emoji / 换行 decode round-trip
    - 已新增风格一致的独立本地前端，明确展示 `gpt-4 → cl100k_base`、raw-text 统计范围、完整 token ids 与最多 300 项逐 token 明细
    - 已为无效/空洞 token id、输入限制、API 状态码、静态目录穿越、CSP/no-store、响应式布局与启动脚本添加回归测试
    - 已固定并验证 `tiktoken==0.14.0`，记录首次加载 encoding 数据需要联网及 `TIKTOKEN_CACHE_DIR` 缓存约定
    - 当前可运行 variant 已扩展为 `basic_bpe`、`split_bpe` 与 `tiktoken_gpt4`；上文 1～3 仍由 `basic_bpe` 存档复现

6. 自研 GPT-4 tokenizer（exercise Step 1～4）
    - 已在 `tokenizers/own_gpt4/` 实现可训练的 `BasicTokenizer` 与 GPT-4 regex `RegexTokenizer`，包含确定性 `train()`、encode 和 decode
    - 已从 `cl100k_base` ranks 自行恢复 100,000 条 merges 与 byte shuffle；实际编解码不委托 tiktoken
    - 已支持 5 个常见 cl100k special tokens，以及默认拒绝、显式允许和普通字面量三类策略
    - 已提供 JSON CLI、双模式独立本地前端和中文使用/测试文档；页面展示真实 special/regex pieces、merge、token bytes 与 golden 对照
    - 已通过 45 项核心测试、16 项前端测试及三个既有 variant 的完整回归测试
    - 当前可运行 variant 已扩展为 `basic_bpe`、`split_bpe`、`tiktoken_gpt4` 与 `own_gpt4`
