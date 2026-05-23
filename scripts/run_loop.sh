#!/bin/bash
# NeuroFlow 全自动学习闭环 — 调度器
# 用法:
#   手动运行: bash scripts/run_loop.sh
#   配置cron: crontab -e
#     0 */5 * * * /mnt/d/neuroflow-model/scripts/run_loop.sh
#
# 循环: 学习 → 训练 → 评估 → 优化 → 准备
# 周期: 每5小时完整一轮

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="/home/administrator/.hermes/hermes-agent/venv/bin/python3"
LOOP_SCRIPT="$SCRIPT_DIR/learn_loop.py"
LOG_FILE="/home/administrator/.hermes/learn_loop_scheduler.log"

echo "[$(date)] ⏰ NeuroFlow 学习闭环触发" >> "$LOG_FILE"

cd "$PROJECT_DIR" || exit 1

# 检查Hermes上下文(web_search)是否可用
# 如果从cron运行, web_search不可用, 只做评估+优化
# 如果从Hermes终端运行, 执行完整闭环

# 1. 确保daemon在运行
DAEMON_PID=$(pgrep -f "daemon_v3.py" | head -1)
if [ -z "$DAEMON_PID" ]; then
    echo "  Daemon未运行, 启动..." >> "$LOG_FILE"
    nohup "$VENV_PYTHON" daemon_v3.py >> daemon_v3.log 2>&1 &
    echo "  Daemon已启动 PID=$!" >> "$LOG_FILE"
else
    echo "  Daemon运行中 PID=$DAEMON_PID" >> "$LOG_FILE"
fi

# 2. 运行学习循环
echo "  运行学习闭环..." >> "$LOG_FILE"
if command -v hermes &>/dev/null; then
    # 通过Hermes CLI触发（如果有hermes web_search权限）
    # hermes -z "run the learn loop script" ...
    # 对于cron模式, 直接运行评估+优化部分（跳过需要web_search的学习）
    "$VENV_PYTHON" -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from learn_loop import load_state, phase_train_wait, phase_evaluate, phase_optimize, phase_prepare, log
state = load_state()
log('📋 Cron模式: 评估+优化 (跳过学习, 需web_search)')
daemon_ok = phase_train_wait(state)
eval_result = phase_evaluate(state)
phase_optimize(state, eval_result)
phase_prepare(state, eval_result)
" >> "$LOG_FILE" 2>&1
else
    "$VENV_PYTHON" "$LOOP_SCRIPT" >> "$LOG_FILE" 2>&1
fi

EXIT_CODE=$?
echo "  退出码: $EXIT_CODE" >> "$LOG_FILE"
echo "  完成: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

exit $EXIT_CODE
