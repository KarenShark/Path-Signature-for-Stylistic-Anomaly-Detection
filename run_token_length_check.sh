#!/bin/bash
# 后台运行 token 长度统计
# 用法: ./run_token_length_check.sh
# 或: bash run_token_length_check.sh

cd "$(dirname "$0")"
PYTHON_BIN="$(pwd)/path_signature_env/bin/python"

echo "开始在后台运行 token 长度统计..."
echo "日志: token_length_stats.log"
echo "查看进度: tail -f token_length_stats.log"
echo ""

nohup "$PYTHON_BIN" check_token_lengths.py > token_length_stats.log 2>&1 &
echo "PID: $!"
echo "运行 'tail -f token_length_stats.log' 查看进度"
