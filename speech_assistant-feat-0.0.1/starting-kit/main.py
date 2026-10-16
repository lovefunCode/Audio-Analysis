# -*- coding: utf-8 -*-
"""
演讲助手评测服务 - Starting Kit (Mock Server)
=========================================================
模拟服务端，提供演讲改进建议生成的流式接口：
  1. POST /generate_advice   流式接口：接收音频文件（multipart 上传）与 query，
                             以 SSE 文本流逐块返回演讲改进建议
  2. GET  /ready             健康检查

SSE 事件约定：
  - event: advice_start   data: {"item_id": ..., "query": ...}
  - event: advice         data: {"append": "<增量建议文本>"}
  - event: done           data: {"item_id": ...}
  - event: error          data: {"message": ...}

环境依赖：Python 3.8+, Flask
"""
import os
import sys
import json
import time
import random
import threading

from flask import Flask, request, Response

if sys.version_info[0] >= 3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json; charset=utf-8"

# =========================================================
# 配置（均可通过环境变量覆盖）
# =========================================================
# 建议流式输出时，每个文本块之间的间隔（秒），模拟大模型逐步输出
ADVICE_CHUNK_DELAY = float(os.getenv("ADVICE_CHUNK_DELAY", "0.2"))

# 模拟的音频转写/分析耗时（秒），模拟首字延迟
ADVICE_FIRST_CHUNK_DELAY = float(os.getenv("ADVICE_FIRST_CHUNK_DELAY", "3"))

# 模拟失败概率（0-1），用于测试失败率统计
ADVICE_FAILURE_RATE = float(os.getenv("ADVICE_FAILURE_RATE", "0"))


# =========================================================
# 模拟建议生成
# =========================================================
def _generate_mock_advice(item_id: str, query: str) -> list:
    """生成一份模拟的演讲改进建议（分段文本列表）。"""
    sections = [
        "【整体评价】本次演讲结构完整，表达清晰，但在开场吸引力与互动设计上仍有提升空间。",
        "【语速与节奏】整体语速偏平稳，建议在关键结论处适当放慢并停顿，给听众留出思考时间；"
        "中段部分语速略快，可通过分段呼吸来控制节奏。",
        "【声音表现】音量层次较为单一，重点内容可通过提高音量或变化音调来强调；"
        "注意避免句尾音量渐弱导致的表达不完整感。",
        "【内容组织】建议采用『结论先行』的表达方式，先给出核心观点，再展开论据；"
        "各要点之间增加过渡句，让逻辑衔接更自然。",
        "【互动与感染力】可通过提问、举手互动等方式增强与听众的连接；"
        "适当加入个人故事或案例，提升演讲的感染力和记忆点。",
        "【改进练习建议】每天进行 5 分钟即兴表达练习，录制后回听并标记问题；"
        "针对本次暴露的语速问题，尝试用节拍器辅助控制语速训练。",
    ]
    return sections


def _iter_advice_chunks(sections: list):
    """将建议文本切分为若干块，模拟大模型逐块输出。

    yield 形式为 (chunk_text, is_last)。
    """
    for i, section in enumerate(sections):
        is_last = (i == len(sections) - 1)
        yield section, is_last


def _stream_advice(item_id: str, query: str, audio_filename: str):
    """流式生成演讲改进建议：以 SSE 文本流逐块输出。

    模拟真实场景：先消耗一段首字延迟（音频转写/分析），再逐块输出建议文本。
    """
    # 模拟失败注入（用于测试失败率统计）
    if ADVICE_FAILURE_RATE > 0 and random.random() < ADVICE_FAILURE_RATE:
        time.sleep(1)
        yield f"event: error\ndata: {json.dumps({'item_id': item_id, 'message': '模拟生成失败'}, ensure_ascii=False)}\n\n"
        return

    # 模拟首字延迟：音频上传后需要先转写/分析
    time.sleep(ADVICE_FIRST_CHUNK_DELAY)

    # 首事件：advice_start
    payload = {"item_id": item_id, "query": query, "audio_filename": audio_filename}
    yield f"event: advice_start\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    sections = _generate_mock_advice(item_id, query)
    for chunk_text, is_last in _iter_advice_chunks(sections):
        payload = {"item_id": item_id, "append": chunk_text}
        yield f"event: advice\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        if is_last:
            break
        time.sleep(ADVICE_CHUNK_DELAY)

    # 结束事件
    yield f"event: done\ndata: {json.dumps({'item_id': item_id}, ensure_ascii=False)}\n\n"


# =========================================================
# 路由
# =========================================================
@app.route('/generate_advice', methods=['POST'])
def generate_advice():
    """（流式接口）接收音频文件（multipart）与 query，SSE 流式返回演讲改进建议。

    请求参数（multipart/form-data）：
      - item_id: 数据项 ID
      - query:   用户的改进诉求（如"帮我听完整音频，给我演讲整体改进建议"）
      - audio:   音频文件（二进制）

    响应：SSE 文本流（text/event-stream），事件见文件头说明。
    """
    try:
        item_id = request.form.get("item_id", "1")
        query = request.form.get("query", "")

        audio_file = request.files.get("audio")
        audio_filename = audio_file.filename if audio_file else ""
        audio_bytes = audio_file.read() if audio_file else b""

        print(f"\n{'='*60}")
        print(f"[流式请求] /generate_advice item_id={item_id} query={query}")
        print(f"[流式请求] audio 文件: {audio_filename} ({len(audio_bytes)/1024/1024:.2f} MB)")
        print(f"{'='*60}")

        if not str(item_id).strip():
            return Response(
                json.dumps({"success": False, "message": "item_id 不能为空"}, ensure_ascii=False),
                status=400, mimetype='application/json; charset=utf-8')
        if not query.strip():
            return Response(
                json.dumps({"success": False, "message": "query 不能为空"}, ensure_ascii=False),
                status=400, mimetype='application/json; charset=utf-8')
        if not audio_bytes:
            return Response(
                json.dumps({"success": False, "message": "音频文件不能为空"}, ensure_ascii=False),
                status=400, mimetype='application/json; charset=utf-8')

        resp = Response(
            _stream_advice(item_id, query, audio_filename),
            status=200,
            mimetype='text/event-stream',
        )
        resp.headers['Cache-Control'] = 'no-cache'
        resp.headers['X-Accel-Buffering'] = 'no'
        resp.headers['Connection'] = 'keep-alive'
        return resp
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(
            json.dumps({"success": False, "message": str(e)}, ensure_ascii=False),
            status=500, mimetype='application/json; charset=utf-8')


@app.route('/ready', methods=['GET'])
def ready():
    """健康检查。"""
    return "True"


# =========================================================
# 启动
# =========================================================
if __name__ == '__main__':
    print("=" * 60)
    print("演讲助手评测模拟服务端 (Starting Kit)")
    print("=" * 60)
    print("接口：")
    print("  POST /generate_advice - 流式：接收音频文件与 query，SSE 流式返回演讲改进建议")
    print("  GET  /ready           - 健康检查")
    print("=" * 60)

    port = int(os.getenv("MOCK_SERVER_PORT", "80"))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
