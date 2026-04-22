#!/bin/bash
set -e

echo "=== H200 语言-算子评测 ==="
echo "测试用例: {{TEST_CASE}} | 卡数: {{CARD_COUNT}}"

# 工作目录固定为 /workspace
cd /workspace

# 执行评测命令
{{TASK_COMMAND}}

echo "=== 评测完成 ==="
