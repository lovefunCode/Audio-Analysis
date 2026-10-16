"""端到端冒烟：python smoke_test.py case.mp3 "帮我听5-10分钟，指出表达节奏问题" [http://localhost:80]
打印 SSE 事件、首字时延、总耗时，并做格式校验。
"""
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.analysis.formatter import validate_output

path = sys.argv[1]
query = sys.argv[2] if len(sys.argv) > 2 else "帮我听并给出改进建议，提升演讲吸引力，指出表达节奏问题"
base = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:80"

t0 = time.time()
first = None
text = []
with open(path, "rb") as f:
    r = requests.post(f"{base}/generate_advice", stream=True, timeout=1800,
                      data={"item_id": "smoke-1", "query": query},
                      files={"audio": (os.path.basename(path), f)})
    r.raise_for_status()
    event = None
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = json.loads(line[5:].strip())
            if event == "advice":
                if first is None:
                    first = time.time() - t0
                text.append(data["append"])
                print(data["append"], end="", flush=True)
            else:
                print(f"\n[{event}] {data}")

full = "".join(text)
print(f"\n--- 首字 {first and first * 1000:.0f}ms，总耗时 {time.time() - t0:.1f}s，{len(full)} 字")
ok, problems = validate_output(full)
print("格式校验:", "通过" if ok else problems)
