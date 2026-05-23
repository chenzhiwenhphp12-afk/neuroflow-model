#!/bin/bash
# NeuroFlow 持续学习同步脚本
# 用法: bash sync_learn.sh
# 从WSL同步KB到HPC，重启HPC daemon

set -e

HPC_USER=acxavb8ge5
HPC_HOST=wuzh02.hpccube.com
HPC_PORT=65091
HPC_KEY=~/.ssh/hpc_key
SSH="ssh -p $HPC_PORT -i $HPC_KEY -o StrictHostKeyChecking=no -o BatchMode=yes"
SCP="scp -P $HPC_PORT -i $HPC_KEY -o StrictHostKeyChecking=no"

WSL_KB=/mnt/d/neuroflow-model/knowledge_base
HPC_NF=/work/home/acxavb8ge5/neuroflow-model

echo "[$(date)] === NeuroFlow 持续学习同步 ==="

# 1. 获取最新KB文件列表（最后50个新增文件）
echo "[1/4] 收集WSL新增KB文件..."
cd "$WSL_KB"
LATEST=$(ls -t *.txt 2>/dev/null | head -50)
COUNT=$(echo "$LATEST" | wc -l)
echo "  最近$COUNT个文件待同步"

# 2. SCP到HPC
echo "[2/4] 同步到HPC..."
echo "$LATEST" | while read f; do
    $SCP "$f" $HPC_USER@$HPC_HOST:$HPC_NF/knowledge_base/ 2>/dev/null
done
echo "  同步完成"

# 3. 取消旧HPC任务，提交新任务
echo "[3/4] 重启HPC daemon..."
$SSH $HPC_USER@$HPC_HOST "scancel -u $HPC_USER 2>/dev/null; sleep 2"
# 统计HPC KB
HPC_COUNT=$($SSH $HPC_USER@$HPC_HOST "ls $HPC_NF/knowledge_base/*.txt 2>/dev/null | wc -l")
echo "  HPC KB: $HPC_COUNT 条"
# 提交新任务
JOB_ID=$($SSH $HPC_USER@$HPC_HOST "cd $HPC_NF && sbatch --parsable nf_hpc_daemon.slurm 2>/dev/null")
echo "  HPC Job: $JOB_ID"

# 4. 交叉验证
echo "[4/4] 交叉验证..."
WSL_VAR=$(tail -1 /mnt/d/neuroflow-model/daemon_v3.log | grep -oP 'var=\K[0-9.]+' || echo "N/A")
WSL_RECON=$(tail -1 /mnt/d/neuroflow-model/daemon_v3.log | grep -oP 'recon=\K[0-9.]+' || echo "N/A")
echo "  WSL: recon=$WSL_RECON var=$WSL_VAR"
echo "  HPC: Job $JOB_ID submitted ($HPC_COUNT KB files)"
echo "[$(date)] === 同步完成 ==="
