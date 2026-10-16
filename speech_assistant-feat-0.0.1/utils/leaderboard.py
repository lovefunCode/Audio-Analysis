# -*- coding: utf-8 -*-
"""
LeaderBoard 平台工具函数。
"""
import os
import json
import traceback

from utils.request import requests_retry_session
from utils.logger import logger

# LeaderBoard API 公共请求头
lb_headers = {"Content-Type": "application/json"}
if os.getenv("LEADERBOARD_API_TOKEN"):
    lb_headers["Authorization"] = "Bearer " + os.getenv("LEADERBOARD_API_TOKEN")


def change_product_unavailable() -> None:
    """
    将当前 submit 标记为产品不可用。

    在校验失败或运行异常时调用，通知 leaderboard 该提交不可用，
    后续排行榜将不展示或降权该提交。
    """
    logger.info("[产品状态] 更改为产品不可用...")
    submit_id = str(os.getenv("SUBMIT_ID", -1))
    try:
        requests_retry_session().post(
            os.getenv(
                "UPDATE_SUBMIT_URL",
                "http://contest.4pd.io:8080/submit/update",
            ),
            data=json.dumps({submit_id: {"product_avaliable": 0}}),
            headers=lb_headers,
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"[产品状态] 标记不可用失败: {e}")


def mark_evaluating(task_id: str) -> None:
    """
    上报任务为 EVALUATING 状态。

    Args:
        task_id: 标注平台任务 ID
    """
    logger.info("[任务状态] 上报 EVALUATING 状态...")
    job_id = os.getenv("JOB_ID") or "-1"
    url = (
        os.getenv(
            "REGISTER_MARK_TASK_URL",
            "http://contest.4pd.io:8080/job/register_mark_task",
        )
        + "/"
        + job_id
    )
    try:
        requests_retry_session().post(
            url,
            data=json.dumps({"task_id": task_id}),
            headers=lb_headers,
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"[任务状态] 上报 EVALUATING 失败: {e}")
