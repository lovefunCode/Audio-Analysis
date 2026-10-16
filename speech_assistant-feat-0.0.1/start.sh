#!/bin/bash

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

# ************************** 脚本执行开始 **************************
log_info "===== 开始执行启动脚本 start.sh ====="

# 打印当前工作目录和 Python 版本
log_info "当前工作目录：$(pwd)"
log_info "当前 Python 版本：$(python --version 2>&1)"

# ************************** 打印指定配置的环境变量 **************************
log_info "===== 开始打印指定配置环境变量 ====="
log_info "时区 TZ：${TZ:-未设置}"
log_info "DARVIN_EVALUATION_OUTPUT_DIR：${DARVIN_EVALUATION_OUTPUT_DIR:-未设置}"
log_info "===== 指定配置环境变量打印完成 ====="

# ************************** 执行 Python 脚本 **************************
log_info "即将启动 Python 脚本：run_for_darvin.py"
python run_for_darvin.py
log_info "Python 脚本：run_for_darvin.py 执行完成"
