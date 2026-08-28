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

server 只监听 127.0.0.1，默认端口是 8008。
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

port="${1:-8008}"
if [[ ! "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
    echo "错误：PORT 必须是 1~65535 之间的整数。" >&2
    exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "错误：没有找到 python3。" >&2
    exit 1
fi

if [[ ! -f "${variant_root}/bpe.py" ]]; then
    echo "错误：没有找到 ${variant_root}/bpe.py。" >&2
    exit 1
fi

if [[ ! -f "${visualizations_dir}/static/index.html" ]]; then
    echo "错误：没有找到 visualizations/static/index.html。" >&2
    exit 1
fi

cd "${workspace_root}"
exec python3 -m tokenizers.basic_bpe.visualizations.server \
    --host 127.0.0.1 \
    --port "${port}"
