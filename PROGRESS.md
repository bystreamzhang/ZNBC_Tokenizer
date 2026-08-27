# PROGRESS

1. Byte pair encoding
    - 已实现 Unicode string -> UTF-8 byte token（`0~255`）
    - 已实现按频率逐次学习 `256, 257, ...` merge token
    - 已支持指定目标 `vocab_size`，以及对新 string 编码
    - 已支持输出 byte/token 长度、节省 token 数、压缩比和可选 token sequence
    - 已明确多个训练样本的边界和相同频率时的确定性规则
    - 已添加单元测试和命令行演示
    - 已将 `bpe.py` 的注释和 docstring 改为更具体的中文说明
    - 已新增集中在 `visualizations/` 的本地浏览器界面；页面通过 Python adapter 调用当前 `bpe.py`，核心算法不依赖展示代码
    - 已定义可复用的页面风格、view/adapter 目录约定和固定启动脚本
