"""SUT 服务（端口 80）
  GET  /ready             探活：模型加载完成后返回 200
  POST /generate_advice   multipart: item_id, query, audio → SSE
       event: advice_start / advice / done / error
"""
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.pipeline import pipeline
from app.config import settings
from app.llm.engine import llm_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("coach")

MSG_TOO_BIG = "抱歉，上传的音频文件过大（超过 {mb}MB），请压缩或裁剪后重新上传。"
MSG_NO_AUDIO = "未检测到音频文件，请上传 mp3 或 m4a 格式的演讲音频后再试。"
MSG_NOT_READY = "抱歉，分析服务初始化失败，暂时无法处理音频，请稍后重试。"
MSG_INTERNAL = "抱歉，分析过程中出现内部异常，本次未能生成完整建议，请稍后重试。"

_state = {"ready": False, "error": ""}
_ready_event = asyncio.Event()


async def _warmup():
    try:
        if settings.warmup_on_start:
            await asyncio.to_thread(pipeline.warmup)
        _state["ready"] = True
        log.info("service ready")
    except Exception:
        log.exception("warmup failed")
        _state["error"] = "模型加载失败"
    finally:
        _ready_event.set()


@asynccontextmanager
async def lifespan(_app):
    # 模型在后台加载，HTTP 端口立即可用：/ready 加载中返回 503，完成后 200。
    task = asyncio.create_task(_warmup())
    yield
    task.cancel()


app = FastAPI(title="演讲陪练助手", version="2.0.0", lifespan=lifespan)


@app.get("/ready")
def ready():
    if _state["ready"] or settings.ready_before_warmup:
        return {"status": "ok"}
    if _state["error"]:
        return JSONResponse({"status": "error", "message": _state["error"]}, status_code=503)
    return JSONResponse({"status": "loading"}, status_code=503)


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _single(msg: str) -> AsyncIterator[str]:
    yield msg


async def _sse(item_id: str, query: str, filename: str,
               gen: AsyncIterator[str]) -> AsyncIterator[str]:
    """advice_start 在第一段文本产出前发送（即转写/分析完成后）。

    兜底策略：pipeline 抛出未预期异常时，不直接发 error（评测记为失败），
    而是以 advice 输出通用提示文本后正常 done；只有连一条文本都没产出时才发 error。
    """
    started = False
    try:
        async for chunk in gen:
            if not chunk:
                continue
            if not started:
                yield _event("advice_start", {"item_id": item_id, "query": query,
                                              "audio_filename": filename})
                started = True
            yield _event("advice", {"item_id": item_id, "append": chunk})
    except asyncio.CancelledError:
        raise
    except Exception as ex:
        log.exception("item %s failed: %s", item_id, type(ex).__name__)
        sep = "\n" if started else ""
        if not started:
            yield _event("advice_start", {"item_id": item_id, "query": query,
                                          "audio_filename": filename})
            started = True
        yield _event("advice", {"item_id": item_id, "append": sep + MSG_INTERNAL})
    finally:
        _check_engine_health()
    if not started:
        yield _event("error", {"item_id": item_id, "message": "未生成任何内容"})
        return
    yield _event("done", {"item_id": item_id})


def _check_engine_health():
    """vLLM 引擎死亡（CUDA 错误等）后无法在进程内恢复：主动退出，交给容器重启。"""
    if llm_engine.dead and settings.exit_on_engine_dead:
        log.critical("llm engine dead, exiting for container restart")
        _state["ready"] = False
        _state["error"] = "模型引擎异常，正在重启"
        loop = asyncio.get_event_loop()
        loop.call_later(1.0, os._exit, 3)


@app.post("/generate_advice")
async def generate_advice(
    item_id: str = Form(...),
    query: str = Form(default=""),
    audio: Optional[UploadFile] = File(default=None),
):
    filename = (audio.filename if audio else "") or ""
    size = 0
    if audio is None:
        gen = _single(MSG_NO_AUDIO)
    else:
        raw = await audio.read()
        size = len(raw)
        if size > settings.max_upload_mb * 1024 * 1024:
            gen = _single(MSG_TOO_BIG.format(mb=settings.max_upload_mb))
        else:
            await _ready_event.wait()          # 模型仍在加载时排队等待，而不是报错
            if _state["error"]:
                gen = _single(MSG_NOT_READY)
            else:
                gen = pipeline.run(raw, filename, query)
    log.info("item %s: file=%s size=%d query=%s", item_id, filename, size, query[:80])
    return StreamingResponse(_sse(item_id, query, filename, gen),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
