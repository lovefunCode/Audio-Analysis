# =========================================================
# 演讲助手评测服务 - Starting Kit Dockerfile
# 模拟服务端
# =========================================================
FROM harbor.4pd.io/lab-platform/inf/python:3.9

LABEL maintainer="wanglimin"
LABEL description="演讲助手评测 Starting Kit (Mock Server)"
LABEL version="0.0.1"

# 安装 curl（健康检查使用）
RUN sed -i 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' /etc/apt/sources.list && \
    sed -i 's|http://security.debian.org/debian-security|http://archive.debian.org/debian-security|g' /etc/apt/sources.list && \
    sed -i '/updates/d' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt --default-timeout=600 -i https://nexus.4pd.io/repository/pypi-all/simple --trusted-host nexus.4pd.io

ADD . /workspace

RUN chmod +x /workspace/start.sh

CMD ["/workspace/start.sh"]
