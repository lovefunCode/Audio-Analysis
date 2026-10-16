#!/usr/bin/env bash
# 下载模型到 ./models，运行容器时挂载到 /workspace/models
set -e
cd "$(dirname "$0")"
mkdir -p models
pip install -q modelscope

python3 - <<'EOF'
import os
from modelscope.hub.snapshot_download import snapshot_download

jobs = [
    # ASR：Paraformer-large + VAD + 标点（共约 1.5GB）
    ("iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "models/paraformer-zh"),
    ("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "models/paraformer-zh/vad"),
    ("iic/punc_ct-transformer_cn-en-common-vocab471067-large", "models/paraformer-zh/punc"),
    # LLM：Qwen2.5-14B-Instruct（约 30GB，bf16 显存约 30GB，A100-80G 可容纳）
    ("Qwen/Qwen2.5-14B-Instruct", "models/qwen2.5-14b-instruct"),
]
for mid, dst in jobs:
    if os.path.isdir(dst) and os.listdir(dst):
        print("skip", dst)
        continue
    print("download", mid, "->", dst)
    snapshot_download(mid, local_dir=dst)
print("done")
EOF
ls models
