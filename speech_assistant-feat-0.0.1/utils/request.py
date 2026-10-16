# -*- coding: utf-8 -*-
"""
HTTP 请求工具：带超时控制和自动重试的 Session。
"""
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# 默认超时 2 分钟
DEFAULT_TIMEOUT = 2 * 60  # seconds


class TimeoutHTTPAdapter(HTTPAdapter):
    """自定义 HTTPAdapter，为所有请求注入默认超时时间。"""

    def __init__(self, *args, **kwargs):
        self.timeout = DEFAULT_TIMEOUT
        if "timeout" in kwargs:
            self.timeout = kwargs.pop("timeout")
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        # 若调用方未显式设置 timeout，则使用构造函数中的默认值
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


def requests_retry_session(
    retries: int = 3,
    backoff_factor: float = 1,
    status_forcelist: tuple = (500, 502, 504, 404, 403),
    session: requests.Session = None,
) -> requests.Session:
    """
    创建带重试机制的 HTTP Session。

    Args:
        retries:          最大重试次数
        backoff_factor:   退避因子（重试间隔 = backoff_factor * 2^(retry-1) 秒）
        status_forcelist: 触发重试的 HTTP 状态码集合
        session:          可选的外部 Session 对象（不传则新建）

    Returns:
        配置好重试和超时的 requests.Session
    """
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = TimeoutHTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
