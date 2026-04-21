#!/bin/bash
set -e

echo "=== H200 语言-算子评测 ==="
echo "测试用例: {{TEST_CASE}} | 卡数: {{CARD_COUNT}}"

# 切换到工作目录
cd {{WORK_DIR}}

# 执行评测命令（编译+测试）
{{TASK_COMMAND}}

echo "=== 评测完成 ==="
