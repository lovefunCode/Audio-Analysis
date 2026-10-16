#!/bin/bash

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

# ************************** 开始执行 **************************
log_info "===== 演讲助手评测 Starting Kit 启动 ====="
log_info "当前工作目录：$(pwd)"
log_info "MOCK_SERVER_PORT：${MOCK_SERVER_PORT:-未设置}"

python main.py
