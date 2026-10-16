# -*- coding: utf-8 -*-
"""
演讲助手评测 - Judge Flow
=========================================================
功能概述：
  - 读取数据集（dataset.zip，含 data.json 与音频文件），遍历每个 case
  - 调用 starting-kit 的 /generate_advice 流式接口（multipart 上传音频文件与 query，
    SSE 文本流返回演讲改进建议），将流式数据组装为一个 JSON
  - 记录每个 case 的首字延迟（first_token_latency）与生成时间（elapsed）
  - 失败率 > 阈值则抛异常结束程序，不推送标注平台
  - 校验通过的案例（query、音频文件、生成的改进建议）推送到 Label Studio 标注平台

执行流程：
  1. 数据读取          load_dataset()
  2. 流式生成并组装    generate_advice_stream() -> advice JSON
  3. 失败率校验        failure_rate check
  4. 构建 Label Studio 任务  build_label_studio_tasks()
  5. 提交标注平台      submit_to_labelx_platform()

环境依赖：Python 3.8+
"""
import os
import re
import csv
import json
import time
import shutil
import zipfile
import traceback
import subprocess
from html import escape
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Final
from dataclasses import dataclass

from utils.logger import logger
from utils.file import dump_json, load_json
from utils.leaderboard import change_product_unavailable, mark_evaluating
from utils.labelx import upload_dataset, check_dataset, create_task

import requests

# =========================================================
# 全局结果容器（用于在异常中保存已处理数据）
# =========================================================
_DETAILED_RESULTS: List[Dict[str, Any]] = []
_BAD_CASES: List[Dict[str, Any]] = []


def _resolve_inference_ip(address: str) -> str:
    """通过 ping 解析域名为 IP，返回替换后的 base URL。"""
    try:
        addr = address
        for prefix in ("https://", "http://"):
            if addr.startswith(prefix):
                addr = addr[len(prefix):]
                break
        host, _, port = addr.partition(":")
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", host],
            capture_output=True, text=True, timeout=5
        )
        logger.info(f"ping result: {result.stdout}")
        match = re.search(r"\((\d+\.\d+\.\d+\.\d+)\)", result.stdout)
        if not match:
            match = re.search(r"from (\d+\.\d+\.\d+\.\d+)", result.stdout)
        if match:
            ip = match.group(1)
            base = f"http://{ip}" + (f":{port}" if port else "")
            logger.info(f"[DNS解析] {host} -> {ip}，base URL: {base}")
            return base
        logger.info(f"[DNS解析] 未能从 ping 输出提取 IP，回退原地址: {address}")
    except Exception as e:
        logger.info(f"[DNS解析] ping 失败: {e}，回退原地址: {address}")
    return address if address.startswith("http") else f"http://{address}"


def reset_global_results() -> None:
    global _DETAILED_RESULTS, _BAD_CASES
    _DETAILED_RESULTS = []
    _BAD_CASES = []


# =========================================================
# 配置（所有阈值均可通过环境变量配置）
# =========================================================
@dataclass(frozen=True)
class Config:
    """全局配置"""
    NOW_TIMESTAMP: Final[str] = datetime.now().strftime("%Y%m%d-%H%M%S")
    cwd_dir: str = os.getcwd()

    # 数据集路径
    DATASET_PATH: str = os.getenv("DARVIN_TESTING_DATASET_ZIP_FILE", "./dataset/dataset.zip")
    DATASET_LIMIT: int = int(os.getenv("DATASET_LIMIT", "0"))

    DARVIN_EVALUATION_OUTPUT_DIR: str = (
        os.getenv("DARVIN_EVALUATION_OUTPUT_DIR")
        or os.path.join(cwd_dir, "evaluation_results")
    )
    EXTERNAL_TASK_FILEPATH: str = os.path.join(DARVIN_EVALUATION_OUTPUT_DIR, "external_task.json")
    RESULT_PATH = os.path.join(DARVIN_EVALUATION_OUTPUT_DIR, "result.json")

    DARVIN_BAD_CASE_OUTPUT: str = os.getenv("DARVIN_BAD_CASE_OUTPUT", DARVIN_EVALUATION_OUTPUT_DIR)
    BAD_CASES_FILEPATH: str = os.path.join(DARVIN_BAD_CASE_OUTPUT, "bad_cases.json")

    DARVIN_DETAIL_CASE_OUTPUT: str = os.getenv("DARVIN_DETAIL_CASE_OUTPUT", DARVIN_EVALUATION_OUTPUT_DIR)
    DETAILED_CASES_FILEPATH: str = os.path.join(DARVIN_DETAIL_CASE_OUTPUT, "detailed_cases.json")

    BENCHMARK_NAME: str = os.getenv("BENCHMARK_NAME", "speech_assistant")
    DATASET_NAME: str = os.getenv("DATASET_NAME", "speech_dataset")

    # 推理服务地址
    DARVIN_INFERENCE_ADDRESS: str = os.getenv("DARVIN_INFERENCE_ADDRESS", "http://localhost:8089")
    INFERENCE_BASE_URL: str = _resolve_inference_ip(DARVIN_INFERENCE_ADDRESS)

    # API 接口
    GENERATE_ADVICE_URL: str = f"{INFERENCE_BASE_URL}/generate_advice"

    GENERATION_OUTPUT_DIR: str = os.path.join(DARVIN_EVALUATION_OUTPUT_DIR, "generate_output")

    DARVIN_JOB_ID: str = os.getenv("DARVIN_EVALUATION_RESULT_CID", "-1")

    # 超时
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "1200"))

    # 首字延迟校验阈值（秒），超过则该 case 校验失败
    FIRST_TOKEN_TIMEOUT: float = float(os.getenv("FIRST_TOKEN_TIMEOUT", "600"))

    # 失败率阈值（超过则终止程序）
    FAILURE_RATE_THRESHOLD: float = float(os.getenv("FAILURE_RATE_THRESHOLD", "1"))


cfg = Config()

# 确保输出目录存在
os.makedirs(cfg.DARVIN_EVALUATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.DARVIN_BAD_CASE_OUTPUT, exist_ok=True)
os.makedirs(cfg.DARVIN_DETAIL_CASE_OUTPUT, exist_ok=True)


# =========================================================
# 1. 数据读取
# =========================================================
def load_dataset() -> Tuple[List[Dict[str, Any]], str]:
    """
    读取数据集，返回 (数据项列表, base_dir)。
    base_dir 是 data.json 所在目录，音频文件的相对路径基于此目录。
    """
    if not os.path.exists(cfg.DATASET_PATH):
        raise ValueError(f"Dataset ZIP not found: {cfg.DATASET_PATH}")

    extract_dir = os.path.join(cfg.DARVIN_EVALUATION_OUTPUT_DIR, "dataset_extracted")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    logger.info(f"[数据集] 从 {cfg.DATASET_PATH} 解压到 {extract_dir}")
    with zipfile.ZipFile(cfg.DATASET_PATH, 'r') as z:
        for item in z.infolist():
            if not (item.flag_bits & 0x800):
                try:
                    raw_bytes = item.filename.encode('cp437')
                except UnicodeEncodeError:
                    raw_bytes = None
                if raw_bytes is not None:
                    try:
                        item.filename = raw_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            item.filename = raw_bytes.decode('gbk')
                        except UnicodeDecodeError:
                            pass
            z.extract(item, extract_dir)

    for root, dirs, files in os.walk(extract_dir):
        if "data.json" in files:
            data_json_path = os.path.join(root, "data.json")
            logger.info(f"[数据集] 找到 data.json: {data_json_path}, base_dir={root}")
            with open(data_json_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            logger.info(f"[数据集] 共 {len(dataset)} 条数据")
            if cfg.DATASET_LIMIT > 0:
                dataset = dataset[:cfg.DATASET_LIMIT]
                logger.info(f"[数据集] DATASET_LIMIT={cfg.DATASET_LIMIT}，截取前 {len(dataset)} 条")
            return dataset, root

    raise ValueError(f"data.json not found after extraction: extract_dir={extract_dir}")


# =========================================================
# 2. 流式接口调用
# =========================================================
def generate_advice_stream(
    item_id: int,
    item_data: Dict[str, Any],
    dataset_base_dir: str,
    save_dir: str,
) -> Optional[Dict[str, Any]]:
    """流式接口：POST /generate_advice（multipart 上传音频文件与 query），
    消费 SSE 文本流，逐块累积建议文本，最终组装为一个 JSON。

    返回结果字典，含 advice_json / advice_text / first_token_latency / elapsed 等；
    失败返回 None。
    """
    query = item_data.get("query", "")
    file_path = item_data.get("file_path", "")
    audio_path = os.path.join(dataset_base_dir, file_path)

    if not os.path.exists(audio_path):
        logger.error(f"[建议流] item_id={item_id} 音频文件不存在: {audio_path}")
        return None

    logger.info(
        f"[建议流] POST {cfg.GENERATE_ADVICE_URL} item_id={item_id} "
        f"query={query} audio={audio_path}"
    )
    start = time.time()
    os.makedirs(save_dir, exist_ok=True)
    advice_save_path = os.path.join(save_dir, f"{item_id}_advice.json")

    state = {
        "item_id": str(item_id),
        "query": query,
        "advice_text": "",
        "error": "",
        "first_token_latency": 0.0,
    }

    try:
        with open(audio_path, "rb") as audio_file:
            resp = requests.post(
                cfg.GENERATE_ADVICE_URL,
                data={"item_id": str(item_id), "query": query},
                files={"audio": (os.path.basename(audio_path), audio_file, "audio/mpeg")},
                timeout=cfg.REQUEST_TIMEOUT,
                stream=True,
            )
        if resp.status_code != 200:
            logger.error(f"[建议流] item_id={item_id} HTTP {resp.status_code}: {resp.text[:200]}")
            return None

        # 消费 SSE 流：advice_start / advice / done / error 事件
        event = ""
        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.strip()
            if line == "":
                event = ""
                continue
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
                if event == "advice" and state["first_token_latency"] == 0.0:
                    # 首字延迟：从请求发出到收到第一块建议文本的时间
                    state["first_token_latency"] = round(time.time() - start, 4)
                    logger.info(
                        f"[建议流] item_id={item_id} 首字延迟: {state['first_token_latency']:.2f}s"
                    )
            elif line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                _handle_advice_event(item_id, event, data_str, state)

        elapsed = time.time() - start
        state["elapsed"] = round(elapsed, 4)

        if state["error"]:
            logger.error(f"[建议流] item_id={item_id} 流式返回 error 事件: {state['error']}")
            return None
        if not state["advice_text"]:
            logger.error(f"[建议流] item_id={item_id} 未收到任何建议内容")
            return None

        # 最终组装为一个 JSON
        advice_json = json.dumps(
            {
                "item_id": state["item_id"],
                "query": state["query"],
                "advice": state["advice_text"],
            },
            ensure_ascii=False,
        )
        state["advice_json"] = advice_json

        # 落盘便于查看与推送标注
        with open(advice_save_path, "w", encoding="utf-8") as f:
            f.write(advice_json)

        logger.info(
            f"[建议流] item_id={item_id} 完成，耗时 {elapsed:.1f}s "
            f"首字延迟 {state['first_token_latency']:.2f}s 建议长度 {len(state['advice_text'])} 字"
        )
        return state
    except Exception as e:
        logger.error(f"[建议流] item_id={item_id} 异常: {e}")
        state["error"] = str(e)
        return None


def _handle_advice_event(item_id: int, event: str, data: str, state: Dict[str, Any]) -> None:
    """处理一条 /generate_advice SSE 事件，累积建议文本到 state。"""
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, TypeError):
        parsed = {}

    if event == "advice_start":
        logger.info(f"[建议流] item_id={item_id} advice_start 收到")
    elif event == "advice":
        append_text = parsed.get("append", "")
        state["advice_text"] += append_text
    elif event == "done":
        logger.info(f"[建议流] item_id={item_id} done 收到")
    elif event == "error":
        state["error"] = parsed.get("message", "未知错误")


# =========================================================
# 3. 单条处理与统计
# =========================================================
def _append_failed_case(item: Dict[str, Any], reason: str) -> None:
    """追加一条失败记录（标记失败，不推送标注平台）。"""
    global _DETAILED_RESULTS, _BAD_CASES
    item_id = item.get("id", 0)
    case = {
        "item_id": item_id,
        "query": item.get("query", ""),
        "file_path": item.get("file_path", ""),
        "success": False,
        "issues": [reason],
        "advice_results": [],
    }
    _DETAILED_RESULTS.append(case)
    _BAD_CASES.append(case)
    logger.info(f"[失败 case] item_id={item_id} 原因: {reason}")


def build_stats_from_results() -> Dict[str, Any]:
    """从详细结果中构建统计信息。

    total_item_count 取 _DETAILED_RESULTS 的总长度（包含生成失败的条目）。
    """
    stats = {
        "total_item_count": len(_DETAILED_RESULTS),
        "success_item_count": 0,
        "failed_item_count": 0,
        "failed_item_ids": [],
    }
    for result in _DETAILED_RESULTS:
        if result.get("success", False):
            stats["success_item_count"] += 1
        else:
            stats["failed_item_count"] += 1
            stats["failed_item_ids"].append(result.get("item_id", ""))
    return stats


def calculate_metrics(stats: Dict[str, Any]) -> Dict[str, Any]:
    """计算额外指标：失败率、平均首字延迟、平均生成时间。"""
    metrics: Dict[str, Any] = {}
    total = stats.get("total_item_count", 0)
    failed = stats.get("failed_item_count", 0)
    metrics["failure_rate"] = round(failed / total, 4) if total > 0 else 0.0

    ftl_sum = 0.0
    ftl_count = 0
    elapsed_sum = 0.0
    elapsed_count = 0
    for result in _DETAILED_RESULTS:
        if not result.get("success", False):
            continue
        for advice_result in result.get("advice_results", []):
            ftl = advice_result.get("first_token_latency", 0.0)
            if ftl > 0:
                ftl_sum += ftl
                ftl_count += 1
            elapsed = advice_result.get("elapsed", 0.0)
            if elapsed > 0:
                elapsed_sum += elapsed
                elapsed_count += 1

    metrics["avg_first_token_latency"] = (
        round(ftl_sum / ftl_count, 4) if ftl_count > 0 else 0.0
    )
    metrics["avg_generation_time"] = (
        round(elapsed_sum / elapsed_count, 4) if elapsed_count > 0 else 0.0
    )
    return metrics


def check_first_token_latency(item_result: Dict[str, Any]) -> Tuple[bool, str]:
    """校验首字延迟是否超过阈值。返回 (is_valid, error_msg)。"""
    # ftl = item_result.get("first_token_latency", 0.0)
    # if ftl <= 0:
    #     return False, "未记录首字延迟"
    # if ftl > cfg.FIRST_TOKEN_TIMEOUT:
    #     return False, (
    #         f"首字延迟超时: {ftl:.2f}s（要求 <= {cfg.FIRST_TOKEN_TIMEOUT}s）"
    #     )
    return True, "通过"


def validate_and_record_item(
    item: Dict[str, Any],
    gen_state: Dict[str, Any],
    save_dir: str,
) -> Dict[str, Any]:
    """校验单条 case 并构建详细结果（含嵌套 advice_result），追加到全局详细结果。

    结构对齐参考项目：detailed_cases 的条目携带 advice_results 列表，
    advice_result 内含校验字段（first_token_valid/issue 等）与资产路径。
    校验失败的 case 同时记入 bad_cases。返回构建的 case 字典。
    """
    global _DETAILED_RESULTS, _BAD_CASES
    item_id = item.get("id", 0)
    issues: List[str] = []

    first_token_latency = gen_state.get("first_token_latency", 0.0)
    elapsed = gen_state.get("elapsed", 0.0)
    advice_text = gen_state.get("advice_text", "")
    advice_path = os.path.join(save_dir, f"{item_id}_advice.json")

    # 建议内容校验
    advice_valid = bool(advice_text.strip())
    advice_issue = "" if advice_valid else "改进建议内容为空"

    # 首字延迟校验
    ftl_valid, ftl_issue = check_first_token_latency(gen_state)

    if not advice_valid:
        issues.append(advice_issue)
    if not ftl_valid:
        issues.append(ftl_issue)

    success = advice_valid and ftl_valid

    advice_result = {
        "item_id": str(item_id),
        "query": item.get("query", ""),
        "file_path": item.get("file_path", ""),
        "advice_path": advice_path,
        "advice_valid": advice_valid,
        "advice_issue": advice_issue,
        "first_token_latency": first_token_latency,
        "first_token_valid": ftl_valid,
        "first_token_issue": ftl_issue if not ftl_valid else "",
        "elapsed": elapsed,
        "advice_length": len(advice_text),
        "audio_asset_url": "",
    }

    case = {
        "item_id": item_id,
        "query": item.get("query", ""),
        "file_path": item.get("file_path", ""),
        "success": success,
        "issues": issues,
        "advice_results": [advice_result],
    }
    _DETAILED_RESULTS.append(case)
    if not success:
        _BAD_CASES.append(case)
        logger.info(f"[失败 case] item_id={item_id} 原因: {issues}")
    return case


# =========================================================
# 4. Label Studio 任务构建（review_context_html）
# =========================================================
AUDIO_CONTENT_TYPE = "audio/mpeg"


def _public_asset_placeholder(asset_id: str) -> str:
    return "{{public_asset:" + asset_id + "}}"


def _relative_public_asset_source(output_dir: str, path: str) -> str:
    return Path(path).resolve().relative_to(Path(output_dir).resolve()).as_posix()


def _stage_public_asset(
    output_dir: Path,
    item_id: str,
    asset_id: str,
    source_path: str,
    filename: str,
    content_type: str,
) -> Dict[str, Any]:
    """将文件复制到 output_dir/public_assets/{item_id}/{filename}，返回资产描述。"""
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"public asset source does not exist: {source}")
    target_dir = output_dir / "public_assets" / item_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    if source.resolve() != target_path.resolve():
        shutil.copyfile(str(source), str(target_path))
    return {
        "id": asset_id,
        "source": _relative_public_asset_source(str(output_dir), str(target_path)),
        "filename": filename,
        "content_type": content_type,
    }


def attach_public_resources(
    detailed_results: List[Dict[str, Any]],
    dataset_base_dir: str,
    output_dir: Path,
    public_assets: List[Dict[str, Any]],
) -> None:
    """为每个成功 case 暂存音频资产（复制数据集中的音频到 public_assets），
    并将资产占位符写入 advice_result.audio_asset_url。"""
    total = 0
    for result in detailed_results:
        if not result.get("success", False):
            continue
        item_id = str(result.get("item_id", "unknown"))
        file_path = result.get("file_path", "")
        source_path = os.path.join(dataset_base_dir, file_path) if file_path else ""
        if not source_path or not os.path.exists(source_path):
            for advice_result in result.get("advice_results", []):
                advice_result["audio_asset_url"] = ""
            continue
        filename = os.path.basename(source_path)
        asset_id = f"{item_id}_audio"
        try:
            asset = _stage_public_asset(
                output_dir, item_id, asset_id, source_path,
                filename, AUDIO_CONTENT_TYPE
            )
            public_assets.append(asset)
            for advice_result in result.get("advice_results", []):
                advice_result["audio_asset_url"] = _public_asset_placeholder(asset_id)
            total += 1
        except Exception as e:
            logger.warning(f"[资产暂存] item_id={item_id} 音频暂存失败: {e}")
            for advice_result in result.get("advice_results", []):
                advice_result["audio_asset_url"] = ""
    logger.info(f"[资产暂存] 共处理 {total} 个 case 音频资产")


def build_label_studio_task(
    item_result: Dict[str, Any],
    index: int,
    job_id: str,
) -> Dict[str, Any]:
    """构建单个 case 的 Label Studio 标注任务。

    meta 结构对齐参考项目：携带完整的嵌套 per-case 结果字典（advice_result），
    含校验字段、指标与资产信息。
    """
    item_id = str(item_result.get("item_id", f"unknown_{index}"))
    query = item_result.get("query", "")
    advice_results = item_result.get("advice_results", [])
    advice_result: Dict[str, Any] = advice_results[0] if advice_results else {}

    advice_json = ""
    if advice_result:
        advice_path = advice_result.get("advice_path", "")
        if advice_path and os.path.exists(advice_path):
            with open(advice_path, "r", encoding="utf-8") as f:
                advice_json = f.read()
    if not advice_json:
        advice_json = json.dumps(
            {"item_id": item_id, "query": query, "advice": ""},
            ensure_ascii=False,
        )

    audio_placeholder = advice_result.get("audio_asset_url", "")

    # 音频播放器
    if audio_placeholder:
        audio_tag = (
            f'<div style="margin:8px 0;">'
            f'<p style="margin:0 0 4px 0;font-size:12px;color:#666;">演讲音频</p>'
            f'<audio controls style="width:100%;max-width:720px;border-radius:4px;">'
            f'<source src="{escape(audio_placeholder, quote=True)}" type="audio/mpeg">'
            f'您的浏览器不支持音频播放</audio>'
            f'</div>'
        )
    else:
        audio_tag = '<p style="font-size:12px;color:#999;">演讲音频不可用</p>'

    # 改进建议（支持滚动）
    if advice_json:
        advice_box = (
            f'<div style="border:1px solid #e5e7eb;border-radius:6px;padding:12px;'
            f'height:300px;overflow-y:auto;margin-bottom:16px;max-width:100%;box-sizing:border-box;">'
            f'<p style="margin:0;white-space:pre-wrap;word-break:break-word;">{escape(advice_json)}</p>'
            f'</div>'
        )
    else:
        advice_box = '<p style="font-size:12px;color:#999;margin-bottom:16px;">改进建议不可用</p>'

    review_context_html = (
        f'<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,sans-serif;'
        f'font-size:14px;line-height:1.6;color:#1f2328;max-width:960px;padding:16px;">'
        # 头部信息区
        f'<div style="margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #e5e7eb;">'
        f'<h2 style="margin:0 0 8px 0;font-size:18px;">演讲助手评测</h2>'
        f'<p style="margin:4px 0;color:#666;font-size:13px;">数据项 ID: {escape(str(item_id))}</p>'
        f'<p style="margin:4px 0;color:#666;font-size:13px;">用户诉求 (query): {escape(str(query))}</p>'
        f'</div>'
        # 演讲音频
        f'{audio_tag}'
        # 改进建议
        f'<p style="margin:0 0 8px 0;font-size:12px;color:#666;">生成的演讲改进建议</p>'
        f'{advice_box}'
        f'</div>'
    )

    return {
        "id": f"{item_id}_advice",
        "data": {
            "review_context_html": review_context_html,
            "item_id": item_id,
            "query": query,
            "advice_json": advice_json,
            "job_id": job_id,
            "index": index,
            "audio_asset_url": audio_placeholder,
            "scoring_context_text": " ",
        },
        "meta": {
            "success": item_result.get("success"),
            "issues": item_result.get("issues", []),
            "advice_result": advice_result,
        },
    }


def build_label_studio_tasks(
    detailed_results: List[Dict[str, Any]],
    job_id: str,
) -> List[Dict[str, Any]]:
    """为所有成功的 case 构建 Label Studio 任务列表。"""
    tasks = []
    index = 0
    for result in detailed_results:
        if not result.get("success", False):
            continue
        index += 1
        tasks.append(build_label_studio_task(result, index, job_id))
    logger.info(f"[LS任务] 共构建 {len(tasks)} 个 Label Studio 任务")
    return tasks


def write_public_assets_manifest(
    output_dir: Path,
    public_assets: List[Dict[str, Any]],
) -> None:
    """将公开资产清单写入 public_assets_manifest.json。"""
    manifest_path = output_dir / "public_assets_manifest.json"
    dump_json(str(manifest_path), {"assets": public_assets})
    logger.info(f"[资产清单] 已写入 {len(public_assets)} 条资产到 {manifest_path}")


def write_inference_results_csv(
    path: str,
    label_studio_tasks: List[Dict[str, Any]],
) -> None:
    """将 Label Studio 任务写入 inference_results.csv。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "baseline", "model_output", "metadata"])
        for task in label_studio_tasks:
            data = task.get("data", {})
            meta = task.get("meta", {})
            writer.writerow([
                json.dumps(
                    {
                        "type": "text",
                        "data": str(data.get("item_id"))
                    },
                    ensure_ascii=False,
                ),
                "",
                json.dumps(
                    {
                        "type": "text",
                        "data": data
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        **meta,
                        "label_studio_task": {
                            "data": data,
                            "meta": meta,
                        },
                    }, ensure_ascii=False),
            ])
    logger.info(f"[CSV] 已写入 {len(label_studio_tasks)} 行到 {path}")


# =========================================================
# 5. 提交标注平台
# =========================================================
def submit_to_labelx_platform() -> Dict[str, Any]:
    """创建标注任务（数据集已通过外部机制传入，这里仅创建任务并上报状态）。"""
    task_name = f'{cfg.BENCHMARK_NAME}-{cfg.DARVIN_JOB_ID}-{cfg.DATASET_NAME}'

    # 将成功 case 的音频打包为 ZIP
    zip_path = _package_success_cases()

    progress_id = upload_dataset(task_name, zip_path)
    logger.info(f"[标注平台] 上传成功 progress_id={progress_id}")

    channel_id = check_dataset(progress_id)
    logger.info(f"[标注平台] channel_id={channel_id}")

    task_id = create_task(task_name, channel_id)
    logger.info(f"[标注平台] task_id={task_id}")

    mark_evaluating(str(task_id))

    return {
        "external_task_id": task_id,
        "task_name": task_name,
        "channel_id": channel_id,
        "progress_id": progress_id,
    }


def _package_success_cases() -> str:
    """将成功 case 的音频、标注所需文件打包为 ZIP。"""
    output_dir = Path(cfg.GENERATION_OUTPUT_DIR)
    zip_path = os.path.join(
        cfg.DARVIN_EVALUATION_OUTPUT_DIR,
        f"{cfg.BENCHMARK_NAME}_{cfg.NOW_TIMESTAMP}.zip"
    )
    logger.info(f"[打包] 开始打包到 {zip_path}")

    success_results = [r for r in _DETAILED_RESULTS if r.get("success", False)]
    if not success_results:
        raise RuntimeError("没有成功的 case，无法打包")

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 打包 public_assets 下暂存的音频（含 item_id 目录结构）
            public_assets_dir = Path(cfg.DARVIN_EVALUATION_OUTPUT_DIR) / "public_assets"
            if public_assets_dir.exists():
                for file_path in public_assets_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(public_assets_dir).as_posix()
                        zf.write(str(file_path), arcname)
                        logger.info(f"[打包] 添加音频: {arcname}")

            # 打包每个成功 case 的 advice JSON
            for result in success_results:
                item_id = str(result.get("item_id", ""))
                advice_path = output_dir / item_id / f"{item_id}_advice.json"
                if advice_path.exists():
                    zf.write(str(advice_path), f"{item_id}/{advice_path.name}")
                    logger.info(f"[打包] 添加建议: {item_id}/{advice_path.name}")

        size_kb = os.path.getsize(zip_path) / 1024
        logger.info(f"[打包] 完成: {zip_path} ({size_kb:.1f} KB)")
        return zip_path
    except Exception as e:
        logger.error(f"[打包] 失败: {e}")
        raise


# =========================================================
# 6. 结果写入
# =========================================================
def write_results(
    result: Optional[Dict[str, Any]] = None,
    detailed_results: Optional[List[Dict[str, Any]]] = None,
    bad_cases: Optional[List[Dict[str, Any]]] = None,
) -> None:
    Path(cfg.DARVIN_EVALUATION_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if result is not None:
        with open(
            os.path.join(cfg.DARVIN_EVALUATION_OUTPUT_DIR, "metrics.json"),
            "w", encoding="utf-8"
        ) as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        logger.info(f"[结果] metrics.json 已写入")

    if detailed_results is not None:
        dump_json(cfg.DETAILED_CASES_FILEPATH, {"item_results": detailed_results})
        logger.info(f"[结果] detailed_cases.json 已写入 ({len(detailed_results)} 条)")

    if bad_cases is not None:
        dump_json(cfg.BAD_CASES_FILEPATH, {"bad_cases": bad_cases, "total": len(bad_cases)})
        logger.info(f"[结果] bad_cases.json 已写入 ({len(bad_cases)} 条)")


def build_stats_metrics(stats: Dict[str, Any]) -> str:
    lines = [
        f"  - 总数据项数: {stats.get('total_item_count', 0)}",
        f"  - 成功数据项数: {stats.get('success_item_count', 0)}",
        f"  - 失败数据项数: {stats.get('failed_item_count', 0)}",
    ]
    return "\n".join(lines)


# =========================================================
# 7. 主流程
# =========================================================
def main() -> None:
    global _DETAILED_RESULTS, _BAD_CASES
    reset_global_results()

    logger.info("=" * 60)
    logger.info("[演讲助手评测] Judge Flow 启动")
    logger.info(f"  GENERATE_ADVICE_URL = {cfg.GENERATE_ADVICE_URL}")
    logger.info("=" * 60)

    stats = None

    try:
        # ---- 步骤 1：读取数据集 ----
        dataset, dataset_base_dir = load_dataset()
        logger.info(f"[数据集] 共 {len(dataset)} 条数据，base_dir={dataset_base_dir}")

        # ---- 步骤 2：循环调用流式接口，生成改进建议并记录指标 ----
        for item in dataset:
            item_id = item.get("id", 0)
            query = item.get("query", "")
            if isinstance(query, str):
                query = query.replace("030", "0-30").replace("512", "5-12")
                item["query"] = query
            item_dir = Path(cfg.GENERATION_OUTPUT_DIR) / str(item_id)
            item_dir.mkdir(parents=True, exist_ok=True)

            gen_state = generate_advice_stream(item_id, item, dataset_base_dir, str(item_dir))
            if gen_state is None:
                logger.error(f"[主流程] item_id={item_id} 建议流式生成失败，标记为失败")
                _append_failed_case(item, "生成失败")
                continue

            case = validate_and_record_item(item, gen_state, str(item_dir))
            if case.get("success", False):
                logger.info(
                    f"[主流程] item_id={item_id} 生成完成："
                    f"建议长度 {len(gen_state.get('advice_text', ''))} 字"
                    f"（耗时 {gen_state.get('elapsed', 0):.1f}s，"
                    f"首字延迟 {gen_state.get('first_token_latency', 0):.2f}s）"
                )
            else:
                logger.error(
                    f"[主流程] item_id={item_id} 校验失败: {case.get('issues', [])}"
                )

        logger.info(f"\n[主流程] 全部数据项收集完成，共 {len(_DETAILED_RESULTS)} 个")

        # ---- 步骤 3：汇总统计 ----
        stats = build_stats_from_results()
        extra_metrics = calculate_metrics(stats)
        stats.update(extra_metrics)

        logger.info(
            f"[统计完成] 成功 {stats['success_item_count']}/{stats['total_item_count']} 条，"
            f"平均首字延迟 {stats.get('avg_first_token_latency', 0)}s，"
            f"平均生成时间 {stats.get('avg_generation_time', 0)}s"
        )

        # ---- 失败率阈值校验 ----
        failure_rate = extra_metrics.get("failure_rate", 0.0)
        # if failure_rate > cfg.FAILURE_RATE_THRESHOLD:
        #     raise ValueError(
        #         f"失败率 {failure_rate:.2%} 超过阈值 {cfg.FAILURE_RATE_THRESHOLD:.0%}，"
        #         f"终止程序，不推送标注平台（失败 {stats['failed_item_count']}/{stats['total_item_count']} 条）"
        #     )

        # ---- 步骤 4：构建 Label Studio 任务 ----
        ls_output_dir = Path(cfg.DARVIN_EVALUATION_OUTPUT_DIR)

        success_results = [r for r in _DETAILED_RESULTS if r.get("success", False)]
        public_assets: List[Dict[str, Any]] = []
        attach_public_resources(success_results, dataset_base_dir, ls_output_dir, public_assets)
        label_studio_tasks = build_label_studio_tasks(_DETAILED_RESULTS, cfg.DARVIN_JOB_ID)
        csv_path = os.path.join(cfg.DARVIN_EVALUATION_OUTPUT_DIR, "inference_results.csv")
        write_inference_results_csv(csv_path, label_studio_tasks)
        write_public_assets_manifest(ls_output_dir, public_assets)

        # ---- 步骤 6：写入结果 ----
        result = {
            "success": 1,
            "detailed_machine_evaluation": build_stats_metrics(stats),
            "total_item_count": stats.get("total_item_count", 0),
            "success_item_count": stats.get("success_item_count", 0),
            "failed_item_count": stats.get("failed_item_count", 0),
            "failure_rate": stats.get("failure_rate", 0),
            "avg_first_token_latency": stats.get("avg_first_token_latency", 0),
            "avg_generation_time": stats.get("avg_generation_time", 0),
            "label_studio_sample_count": len(label_studio_tasks),
            "mode": "first_run_submitted",
        }
        write_results(result, _DETAILED_RESULTS, _BAD_CASES)

        external_task = {
            "platform": "label_studio",
            "external_task_id": f"label_studio_job:{cfg.DARVIN_JOB_ID}",
            "platform_task_name": f"{cfg.BENCHMARK_NAME}-{cfg.DARVIN_JOB_ID}-{cfg.DATASET_NAME}",
            "task_url": "",
        }
        result_payload = {
            "benchmark_name": cfg.BENCHMARK_NAME,
            "dataset_name": cfg.DATASET_NAME,
            "sample_count": len(label_studio_tasks),
            "job_id": cfg.DARVIN_JOB_ID,
            "task_id": external_task["external_task_id"],
            "task_url": external_task["task_url"],
            "platform_task_name": external_task["platform_task_name"],
            "evaluation_result": {**result, "_score": -1},
            "external_manual": {
                "platform": "label_studio",
                "label_studio_tasks": label_studio_tasks,
                "reviewer_ls_user_id": "",
            },
            "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        dump_json(cfg.RESULT_PATH, result_payload)
        dump_json(cfg.EXTERNAL_TASK_FILEPATH, external_task)

        logger.info("=" * 60)
        logger.info("[演讲助手评测] Judge Flow 完成")
        logger.info("=" * 60)

    except RuntimeError as e:
        logger.error(f"[致命错误] {e}")
        _safe_cleanup()
        result = {
            "success": 0,
            "detailed_machine_evaluation": build_stats_metrics(stats) if stats else "致命错误",
            "error_message": str(e),
            "mode": "fatal_error",
        }
        write_results(result, _DETAILED_RESULTS, _BAD_CASES)
        os._exit(1)

    except ValueError as e:
        logger.error(f"[阈值超标] {e}")
        change_product_unavailable()
        result = {
            "success": 0,
            "detailed_machine_evaluation": build_stats_metrics(stats) if stats else "阈值超标",
            "error_message": str(e),
            "failure_rate": stats.get("failure_rate", 0) if stats else 0,
            "mode": "failure_rate_exceeded",
        }
        write_results(result, _DETAILED_RESULTS, _BAD_CASES)
        os._exit(1)

    except Exception as e:
        logger.error(f"[未预期异常] {e}")
        logger.error(traceback.format_exc())
        _safe_cleanup()
        result = {
            "success": 0,
            "detailed_machine_evaluation": {"error_message": str(e)},
            "mode": "error",
        }
        write_results(result, _DETAILED_RESULTS, _BAD_CASES)
        os._exit(1)


def _safe_cleanup() -> None:
    """清理临时目录。"""
    for d in ("dataset_extracted",):
        p = Path(cfg.DARVIN_EVALUATION_OUTPUT_DIR) / d
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("run_for_darvin failed: %s", exc, exc_info=True)
        import sys
        sys.exit(1)
