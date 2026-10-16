import requests
import time
import os
import json
import hashlib
import logging

logger = logging.getLogger(__file__)

AK = os.getenv("AK") or "Z3IwdWNqcWszaWwzOW9wamJhbGRxZG5nNWJ0N2hlcDU="
SK = os.getenv("SK") or "OGhkbmFudGczY3M2NWJpMnV0Y2puY2k4NDVsNzd0bmM="
LABELX_ADDRESS = os.getenv("LABELX_ADDRESS") or "http://172.26.1.71:8060"
LABELX_USER_ID = int(os.getenv("LABELX_USER_ID") or "103")
LABELX_CHANNEL_TYPE = int(os.getenv("LABELX_CHANNEL_TYPE") or "12")
LABELX_TEMPLATE_ID = int(os.getenv("LABELX_TEMPLATE_ID") or "407")
LABELX_TEMPLATE_TASK_ID = int(os.getenv("LABELX_TEMPLATE_TASK_ID") or "648")
LABELX_GROUP_ID = int(os.getenv("LABELX_GROUP_ID") or "1")
LABELX_DATA_ASSIGN_MODE = int(os.getenv("LABELX_DATA_ASSIGN_MODE") or "1")

# HTTP 请求超时时间（秒）
# 可通过环境变量 LABELX_REQUEST_TIMEOUT 配置
LABELX_REQUEST_TIMEOUT = int(os.getenv("LABELX_REQUEST_TIMEOUT", "6000"))  # 默认 10 分钟


def compute_sign(access_key, timestamp, secret_key):
    data = access_key + str(timestamp) + secret_key
    md5_hash = hashlib.md5(data.encode())
    return md5_hash.hexdigest()


def upload_dataset(name, filepath):
    retries = 600
    res = None
    while retries > 0:
        try:
            url = LABELX_ADDRESS + '/labelx/api/open/api/data/upload'
            current_time = int(time.time())
            para = {
                'accessKey': AK,
                "timestamp": current_time,
                "sign": compute_sign(AK, current_time, SK)
            }
            f = open(filepath, "rb")
            data = {"channelName": name, "channelType": LABELX_CHANNEL_TYPE,
                    "templateId": LABELX_TEMPLATE_ID, "userId": LABELX_USER_ID, "subtitleSplit": 0}
            logger.info(f"请求标注平台：上传数据集 {name} 到 {url} 参数 {para}, data={data}")
            res = requests.post(url, params=para,
                                data=data,
                                files={'file': f},
                                timeout=LABELX_REQUEST_TIMEOUT)
            f.close()
            res = res.json()
            return res['body']['resultObject']['id']
        except Exception as e:
            logger.exception(e)
            logger.error(f"请求标注平台失败：上传数据集失败, {res}。重试中...{retries}")
            time.sleep(10)
            retries -= 1
    assert False, "请求标注平台失败：上传数据集失败"


def check_dataset(id):
    retries = 600
    res = None
    while retries > 0:
        try:
            url = LABELX_ADDRESS + '/labelx/api/open/api/data/upload/status'
            current_time = int(time.time())
            para = {
                'accessKey': AK,
                "timestamp": current_time,
                "sign": compute_sign(AK, current_time, SK),
                "progressId": id
            }
            res = requests.get(url, params=para, timeout=LABELX_REQUEST_TIMEOUT).json()
            if res['body']['uploadStatus'] == 2:
                return res['body']['channelId']
            else:
                logger.info("请求标注平台：检查数据集结果未完成，状态为{} 重试中...{}".format(res, retries))
                time.sleep(10)
                retries -= 1
        except Exception as e:
            logger.exception(e)
            logger.warn(f"请求标注平台失败：检查数据集结果失败, {res}。重试中...{retries}")
            time.sleep(10)
            retries -= 1
    assert False, "请求标注平台：检查数据集结果未完成"


def create_task(name, channel_id):
    retries = 10
    res = None
    while retries > 0:
        try:
            url = LABELX_ADDRESS + '/labelx/api/open/api/task/create'
            current_time = int(time.time())
            para = {
                'accessKey': AK,
                "timestamp": current_time,
                "sign": compute_sign(AK, current_time, SK)
            }
            res = (requests.post(url, params=para, headers={"Content-Type": "application/json"}, data=json.dumps(
                {"taskName": name, "userId": LABELX_USER_ID, "templateTaskId": LABELX_TEMPLATE_TASK_ID,
                 "groupId": LABELX_GROUP_ID, "dataAssignMode": LABELX_DATA_ASSIGN_MODE,
                 "channelId": channel_id}), timeout=LABELX_REQUEST_TIMEOUT).json())
            return res['body']['resultObject']
        except Exception as e:
            logger.exception(e)
            logger.warn(f"请求标注平台失败：创建标注任务, {res}。重试中...{retries}")
            time.sleep(10)
            retries -= 1
    assert False, "请求标注平台：检查数据集结果未完成"


def get_ans(task_id):
    retries = 10
    res = None
    while retries > 0:
        try:
            url = LABELX_ADDRESS + '/labelx/api/open/api/task/data/result'
            current_time = int(time.time())
            para = {
                'accessKey': AK,
                "timestamp": current_time,
                "sign": compute_sign(AK, current_time, SK),
                "taskId": task_id
            }
            res = requests.get(url, params=para, timeout=LABELX_REQUEST_TIMEOUT).json()
            return res['body']['resultList']
        except:
            logger.warn(f"请求标注平台失败：获取标注结果, {res}，重试中...{retries}")
            time.sleep(10)
            retries -= 1
    assert False, "请求标注平台失败：获取标注结果"
