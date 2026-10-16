# vLLM 官方镜像自带 CUDA + torch + vllm，避免版本冲突
FROM vllm/vllm-openai:v0.6.3

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 模型运行时挂载到 /workspace/models（或通过环境变量指向挂载路径）
ENV ASR_MODEL_PATH=/workspace/models/paraformer-zh \
    LLM_MODEL_PATH=/workspace/models/qwen2.5-14b-instruct \
    MODELSCOPE_CACHE=/workspace/models/.cache \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s \
    CMD curl -fs http://127.0.0.1:80/ready || exit 1

ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
