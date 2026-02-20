#!/bin/bash
# 后台运行全量 embeddings 计算（关闭 terminal 后继续运行）
# 用法: ./run_compute_embeddings.sh
# 查看进度: tail -f embeddings.log
# 检查进程: ps aux | grep compute_all_embeddings

cd "$(dirname "$0")"
PYTHON_BIN="$(pwd)/path_signature_env/bin/python"
LOG="$PWD/embeddings.log"
PIDFILE="$PWD/embeddings.pid"

if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "已在运行 (PID: $OLD_PID)"
        echo "查看进度: tail -f $LOG"
        exit 0
    fi
fi

echo "启动 embeddings 计算..."
echo "  日志: $LOG"
echo "  查看进度: tail -f $LOG"
echo ""

nohup "$PYTHON_BIN" -u compute_all_embeddings.py >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "PID: $(cat $PIDFILE)"
echo "进程已后台运行，关闭 terminal 不会中断"
