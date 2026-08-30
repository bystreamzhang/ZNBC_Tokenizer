#!/usr/bin/env bash
set -euo pipefail

visualizations_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
variant_root="$(cd -- "${visualizations_dir}/.." && pwd)"
workspace_root="$(cd -- "${variant_root}/../.." && pwd)"

usage() {
    cat <<'EOF'
用法：
  ./visualizations/run.sh [PORT]

示例：
  ./visualizations/run.sh
  ./visualizations/run.sh 9000

server 只监听 127.0.0.1，默认端口是 8012。
按 Ctrl+C 停止。
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

port="${1:-8012}"
if [[ ! "${port}" =~ ^[0-9]{1,5}$ ]]; then
    echo "错误：PORT 必须是 1~65535 之间的整数。" >&2
    exit 2
fi
port_number=$((10#${port}))
if (( port_number < 1 || port_number > 65535 )); then
    echo "错误：PORT 必须是 1~65535 之间的整数。" >&2
    exit 2
fi

if [[ -x "${variant_root}/.venv/bin/python" ]]; then
    python_bin="${variant_root}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
else
    echo "错误：没有找到 ${variant_root}/.venv/bin/python 或 PATH 中的 python3。" >&2
    exit 1
fi

if [[ ! -f "${variant_root}/tokenizer.py" ]]; then
    echo "错误：没有找到 ${variant_root}/tokenizer.py。" >&2
    exit 1
fi

if [[ ! -f "${visualizations_dir}/static/index.html" ]]; then
    echo "错误：没有找到 visualizations/static/index.html。" >&2
    exit 1
fi

if ! "${python_bin}" -c 'import tiktoken' >/dev/null 2>&1; then
    echo "错误：没有找到 Python 依赖 tiktoken。" >&2
    echo "请先运行：${python_bin} -m pip install -r ${variant_root}/requirements.txt" >&2
    exit 1
fi

cd "${workspace_root}"
exec "${python_bin}" -m tokenizers.tiktoken_gpt4.visualizations.server \
    --host 127.0.0.1 \
    --port "${port_number}"
