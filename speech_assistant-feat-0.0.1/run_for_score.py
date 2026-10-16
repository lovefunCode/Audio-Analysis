# -*- coding: utf-8 -*-
"""
演讲助手评测 - 评分脚本
=========================================================
功能：
  1. 从 Label Studio 拉取人工标注结果
  2. 解析标注数据
  3. 计算各项指标（结合 run_for_darvin 输出的机器指标）
  4. 生成评分报告

算分逻辑：
  - 每个 case 的有效标记：overall_evaluation 必须“有且仅有”
    ["以上问题均未出现"] 一项，否则该 case 直接给 0 分
  - 有效 case 的分数 = model_understand + model_execute_query + out_four_segment
    + four_sagement_contains_good + end_suggest + content_score + language_expression
    + improve_direction + audio_content + audio_segment + content_compliance
  - case 平均分 = 所有 case 的总分 / 数据集中的数据量（metrics.json 的 total_item_count）
  - 失败率 = （数据集中 case 数 - 人工评分大于 0 的 case 数）/ 数据集中 case 数
  - 平均延迟：从 run_for_darvin 输出的 metrics.json 获取 avg_first_token_latency，
    换算为毫秒输出到最终指标
  - 最后将人工标注结果 evaluation_task_answers 的内容写回 detailed_cases.json
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

import requests

from utils.file import dump_json, load_json
from utils.logger import logger

# ---------- 配置 ----------
def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


# 人工判分的打分字段（各为 {number: N} 结构），有效 case 的分数为各字段之和
SCORE_FIELDS = (
    "model_understand",          # 模型理解
    "model_execute_query",       # 模型执行用户诉求
    "out_four_segment",          # 输出四段内容
    "four_sagement_contains_good",  # 四段内容包含优点
    "end_suggest",               # 结尾建议
    "content_score",             # 内容评分
    "language_expression",       # 语言表达
    "improve_direction",         # 改进方向
    "audio_content",             # 音频内容
    "audio_segment",             # 音频分段
    "content_compliance",        # 内容合规
)
# 有效性判定字段与期望值：overall_evaluation 必须“有且仅有”该值
OVERALL_FIELD = "overall_evaluation"
OVERALL_OK_VALUE = "以上问题均未出现"


# ---------- Label Studio 结果拉取 ----------
def extract_result_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Label Studio result response must be a JSON object")
    code = payload.get("code")
    if code not in (None, 200, "200"):
        raise RuntimeError(
            f"Label Studio result pull failed: code={code}, message={payload.get('message', '')}"
        )
    candidates = [payload]
    for key in ("data", "body"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in candidates:
        result_list = candidate.get("resultList")
        if isinstance(result_list, list):
            return [item for item in result_list if isinstance(item, dict)]
    raise ValueError(f"Label Studio result response missing resultList: {payload!r}")


def result_url(external_task: Dict[str, Any]) -> str:
    explicit = str(external_task.get("result_url") or _env("DARVIN_LABEL_STUDIO_RESULT_URL")).strip()
    if explicit:
        return explicit
    base_url = (
        _env("DARVIN_BACKEND_BASE_URL")
        or _env("DARVIN_PUBLIC_BASE_URL")
        or _env("API_PUBLIC_BASE_URL")
    )
    external_task_id = str(external_task.get("external_task_id") or "").strip()
    if not base_url or not external_task_id:
        raise ValueError("result_url or DARVIN_BACKEND_BASE_URL with external_task_id is required")
    return f"{base_url.rstrip('/')}/api/evaluation/external-tasks/{quote(external_task_id, safe='')}/result"


def result_token(external_task: Dict[str, Any]) -> str:
    token = str(
        external_task.get("result_token") or _env("DARVIN_LABEL_STUDIO_RESULT_TOKEN")
    ).strip()
    if not token:
        raise ValueError("result_token is required")
    return token


def pull_label_studio_results(external_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    token = result_token(external_task)
    response = requests.get(
        result_url(external_task),
        headers={
            "Authorization": f"Bearer {token}",
            "X-Darvin-Result-Token": token,
        },
        timeout=int(_env("DARVIN_LABEL_STUDIO_RESULT_TIMEOUT", "120")),
    )
    response.raise_for_status()
    return extract_result_list(response.json())


# ---------- 标注结果解析 ----------
def _extract_number(val: Any) -> float:
    """从 {number: N} 结构或纯数值中提取数值。"""
    if isinstance(val, dict):
        val = val.get("number", 0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _is_valid_case(eta: Dict[str, Any]) -> bool:
    """该 case 有效：overall_evaluation 必须“有且仅有”[以上问题均未出现]。"""
    oe = eta.get(OVERALL_FIELD, [])
    return isinstance(oe, list) and oe == [OVERALL_OK_VALUE]


def _field_score(eta: Dict[str, Any]) -> float:
    """case 分数：overall_evaluation 有且仅有“以上问题均未出现”时为各打分字段之和，否则 0 分。"""
    if not _is_valid_case(eta):
        return 0.0
    return sum(_extract_number(eta.get(field, 0)) for field in SCORE_FIELDS)


def _field_sum(eta: Dict[str, Any], field: str) -> float:
    """指定打分字段的取值（无效 case 按 0 计入）。"""
    if not _is_valid_case(eta):
        return 0.0
    return _extract_number(eta.get(field, 0))


# ---------- 指标计算 ----------
def calculate_metrics(
    label_results: List[Dict[str, Any]],
    metrics_data: Dict[str, Any],
    detailed_cases: Dict[str, Any],
) -> Dict[str, Any]:
    """计算所有指标，并将人工标注结果写回 detailed_cases。"""
    total_item_count = int(metrics_data.get("total_item_count", 0))
    if total_item_count == 0:
        raise ValueError("metrics.json 中 total_item_count 为 0，无法计算指标")

    annotation_map: Dict[str, Dict[str, Any]] = {}
    bad_cases: List[Dict[str, Any]] = []

    # 机器生成成功的 item_id 集合（来自 run_for_darvin 输出的 detailed_cases）
    machine_success_ids = {
        str(case.get("item_id", ""))
        for case in detailed_cases.get("item_results", [])
        if case.get("success")
    }

    sum_field_scores: Dict[str, float] = {field: 0.0 for field in SCORE_FIELDS}
    case_score_sum = 0.0
    success_count = 0

    for item in label_results:
        data = item.get("data", {})
        item_id = str(data.get("item_id", ""))
        eta = item.get("evaluation_task_answers", {})
        if not isinstance(eta, dict) or not eta:
            logger.warning("item_id=%s 无有效标注数据，跳过", item_id)
            continue

        valid = _is_valid_case(eta)
        case_score = _field_score(eta)
        annotation_map[item_id] = {
            "valid": valid,
            "case_score": round(case_score, 4),
            "evaluation_task_answers": eta,
        }

        # 成功 case：机器生成成功 且 人工标注有效 且 人工评分 > 0
        machine_ok = item_id in machine_success_ids
        is_success = machine_ok and valid and case_score > 0

        if is_success:
            success_count += 1
            for field in SCORE_FIELDS:
                sum_field_scores[field] += _field_sum(eta, field)
        else:
            bad_cases.append({
                "item_id": item_id,
                "reason": _build_bad_case_reason(eta),
                "machine_success": machine_ok,
                "valid": valid,
                "case_score": round(case_score, 4),
            })

        case_score_sum += case_score

    # 各字段平均分（分母为人工评分大于 0 的 case 数）
    avg_field_scores = {
        f"平均{field}": round(sum_field_scores[field] / success_count, 4) if success_count else 0.0
        for field in SCORE_FIELDS
    }
    # case 平均分 = 所有 case 总分 / 数据集中的数据量
    avg_case = round(case_score_sum / total_item_count, 4)
    # 失败率 =（数据集中 case 数 - 人工评分大于 0 的 case 数）/ 数据集中 case 数
    failure_rate = round((total_item_count - success_count) / total_item_count, 4) if total_item_count else 0.0
    # 平均首字延迟（秒 -> 毫秒）
    first_token_latency_ms = round(float(metrics_data.get("avg_first_token_latency", 0) or 0) * 1000, 2)

    result: Dict[str, Any] = {
        # **avg_field_scores,
        "case平均分": avg_case,
        "平均首字延迟(ms)": first_token_latency_ms,
        "失败率": failure_rate,
    }
    # _score 供平台读取：取 case 平均分作为综合得分
    result["_score"] = avg_case

    _update_detailed_cases(detailed_cases, annotation_map)

    return result, bad_cases


def _build_bad_case_reason(eta: Dict[str, Any]) -> str:
    oe = eta.get(OVERALL_FIELD, [])
    if isinstance(oe, list) and oe == [OVERALL_OK_VALUE]:
        # overall 有效，但 case 总分为 0
        detail = ", ".join(
            f"{field}={eta.get(field)}"
            for field in SCORE_FIELDS
        )
        return f"总分为 0（评分：{detail}）"
    # overall 不满足“有且仅有以上问题均未出现”
    reason = "overall_evaluation 不满足“有且仅有 以上问题均未出现”"
    if isinstance(oe, list):
        reason += f"（实际: {oe}）"
    return reason


def _update_detailed_cases(
    detailed_cases: Dict[str, Any],
    annotation_map: Dict[str, Dict[str, Any]],
) -> None:
    """将人工标注结果 evaluation_task_answers 写回 detailed_cases。"""
    for item in detailed_cases.get("item_results", []):
        item_id = str(item.get("item_id", ""))
        if item_id in annotation_map:
            item["evaluation_task_answers"] = annotation_map[item_id]["evaluation_task_answers"]
            item["annotation_valid"] = annotation_map[item_id]["valid"]


# ---------- 辅助：加载文件 ----------
def load_external_task(evaluation_dir: Path) -> Dict[str, Any]:
    data_path = _env("DARVIN_AUTO_SCORING_DATA_PATH")
    candidates: List[Path] = []
    if data_path:
        candidates.append(Path(data_path))
    candidates.append(evaluation_dir / "external_task.json")
    for candidate in candidates:
        if candidate.is_file():
            return load_json(str(candidate))
    raise ValueError(f"external_task.json not found in {evaluation_dir}")


def load_metrics(evaluation_dir: Path) -> Dict[str, Any]:
    metrics_path = evaluation_dir / "metrics.json"
    if not metrics_path.is_file():
        raise ValueError(f"metrics.json not found at {metrics_path}")
    data = load_json(str(metrics_path))
    if not isinstance(data, dict):
        raise ValueError("metrics.json is not a JSON object")
    return data


def load_detailed_cases(evaluation_dir: Path) -> Dict[str, Any]:
    detail_case_output = _env("DARVIN_DETAIL_CASE_OUTPUT", str(evaluation_dir))
    cases_path = Path(detail_case_output) / "detailed_cases.json"
    if not cases_path.is_file():
        raise ValueError(f"detailed_cases.json not found at {cases_path}")
    data = load_json(str(cases_path))
    if not isinstance(data, dict):
        raise ValueError("detailed_cases.json is not a JSON object")
    return data


# ---------- 入口 ----------
def main() -> None:
    evaluation_dir = Path(_env("DARVIN_EVALUATION_OUTPUT_DIR", "./evaluation_results"))
    metrics_output = _env("DARVIN_AUTO_SCORING_METRICS_OUTPUT", "./auto_scoring_metrics.json")

    if not evaluation_dir.exists():
        raise ValueError(f"evaluation output dir does not exist: {evaluation_dir}")

    external_task = load_external_task(evaluation_dir)
    if str(external_task.get("platform") or "").lower() != "label_studio":
        raise ValueError(f"unsupported manual result platform: {external_task.get('platform')}")

    metrics_data = load_metrics(evaluation_dir)
    detailed_cases = load_detailed_cases(evaluation_dir)

    raw_results = pull_label_studio_results(external_task)
    logger.info("拉取到 %d 条标注结果", len(raw_results))

    final_result, bad_cases = calculate_metrics(raw_results, metrics_data, detailed_cases)

    dump_json(metrics_output, final_result)
    logger.info("saved auto scoring metrics to %s", metrics_output)

    detail_case_output = _env("DARVIN_DETAIL_CASE_OUTPUT", str(evaluation_dir))
    dump_json(str(Path(detail_case_output) / "detailed_cases.json"), detailed_cases)
    logger.info("updated detailed_cases.json with annotation results")

    bad_case_output = _env("DARVIN_BAD_CASE_OUTPUT", str(evaluation_dir))
    dump_json(str(Path(bad_case_output) / "bad_cases.json"), {"bad_cases": bad_cases, "total": len(bad_cases)})
    logger.info("saved bad_cases.json to %s (%d cases)", str(Path(bad_case_output) / "bad_cases.json"), len(bad_cases))

    logger.info("===== 指标摘要 =====")
    for key, val in final_result.items():
        if key != "_score":
            logger.info("  %s: %s", key, val)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("run_for_score failed: %s", exc, exc_info=True)
        sys.exit(1)
