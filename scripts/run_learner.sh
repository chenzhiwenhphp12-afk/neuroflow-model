#!/bin/bash
# NeuroFlow 自动学习 — 周期性调度入口
# 用法: 
#   手动触发: bash scripts/run_learner.sh
#   添加到cron: crontab -e 添加下面一行:
#     0 * * * * /mnt/d/neuroflow-model/scripts/run_learner.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="/home/administrator/.hermes/hermes-agent/venv/bin/python3"
LOG_FILE="/home/administrator/.hermes/auto_learner_cron.log"

echo "[$(date)] === Auto Learner ===" >> "$LOG_FILE"

# 切换到项目目录
cd "$PROJECT_DIR" || exit 1

# 检查daemon是否在运行（仅日志记录）
if pgrep -f "daemon_v3.py" > /dev/null 2>&1; then
    echo "  Daemon: running" >> "$LOG_FILE"
else
    echo "  Daemon: NOT running (will start)" >> "$LOG_FILE"
    # 如果daemon不在运行，启动它
    nohup "$VENV_PYTHON" daemon_v3.py >> daemon_v3.log 2>&1 &
    echo "  Daemon started PID $!" >> "$LOG_FILE"
fi

# 运行自动学习
"$VENV_PYTHON" "$SCRIPT_DIR/auto_learner.py" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "  Exit: $EXIT_CODE" >> "$LOG_FILE"
echo "  Done: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

exit $EXIT_CODE
