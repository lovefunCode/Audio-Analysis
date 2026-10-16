# =========================================================
# 演讲助手评测 - Judge Flow Dockerfile
# =========================================================
FROM harbor.4pd.io/lab-platform/inf/python:3.9

LABEL maintainer="wanglimin"
LABEL description="演讲助手评测 Judge Flow"
LABEL version="0.0.1"

# 设置工作目录
WORKDIR /workspace

# 优先复制 requirements.txt，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install -r requirements.txt --default-timeout=600 -i https://nexus.4pd.io/repository/pypi-all/simple --trusted-host nexus.4pd.io

# 复制所有文件
ADD . /workspace

# 给 start.sh 赋予执行权限
RUN chmod +x /workspace/start.sh

# 运行 start.sh 脚本
CMD ["/workspace/start.sh"]
