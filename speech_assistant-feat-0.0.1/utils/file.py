import json
from typing import Any


def load_json(path: str, raise_for_invalid: bool = False) -> Any:
    """读取path json文件转为对象"""
    with open(path, "r", encoding="utf-8") as f:
        if raise_for_invalid:

            def parse_constant(s: str):
                raise ValueError("非法json字符: %s" % s)

            return json.load(f, parse_constant=parse_constant)
        return json.load(f)


def dump_json(path: str, obj: Any):
    """将obj对象以json形式写入path文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
