"""评测请求模型 + 评测集逐条校验 + 默认 judge prompt。

纯逻辑, 不做网络/SSH, 可被 test_eval.py 直接导入。
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid")
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# ── 默认 judge prompt (前端可覆盖) ──

DEFAULT_FC_PARAM_PROMPT = (
    "你是函数调用参数评审。给定用户请求、期望参数(gold)与模型实际提取的参数(pred), "
    "评估 pred 是否正确表达了用户意图并与 gold 语义一致(容许等价写法, 如 '15:00' 与 '下午3点')。"
    "只输出 JSON: {\"score\": 1-5, \"reason\": \"简述\"}。5=完全正确, 1=完全错误。"
)

DEFAULT_SUBJECTIVE_PROMPT = (
    "你是主观任务评审。给定用户问题、参考答案(reference, 可能为空)与模型回答(answer), "
    "综合正确性、完整性、相关性打分。只输出 JSON: {\"score\": 1-5, \"reason\": \"简述\"}。"
)


class ModelUnderTest(BaseModel):
    model_config = _STRICT
    name: str                                    # 报告/产物命名, 安全字符
    model_name_or_path: str                      # 服务器上基座路径
    adapter_path: Optional[str] = None           # 微调 adapter 目录 (可空=纯基座)
    template: str = "qwen3_5_nothink"


class EvalRequest(BaseModel):
    model_config = _STRICT
    models: list[ModelUnderTest] = Field(min_length=1)
    task_types: list[Literal["function_call", "subjective"]] = Field(min_length=1)
    gpus: str = ""
    api_port: int = 8000
    ready_timeout: int = 600                      # 端点就绪等待上限(秒)
    fc_param_prompt: str = DEFAULT_FC_PARAM_PROMPT
    subjective_prompt: str = DEFAULT_SUBJECTIVE_PROMPT

    def validate_names(self) -> None:
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError("模型 name 不能重复")
        for n in names:
            if not _NAME_RE.match(n):
                raise ValueError(f"模型 name 非法: {n} (仅限字母数字 . _ -)")


def validate_model_name(name: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise ValueError("模型 name 非法")
    return name


# ── 评测集逐条校验 ──

def _validate_tags(item: dict[str, Any], item_id: str) -> None:
    tags = item.get("tags")
    if tags is None:
        return
    if not isinstance(tags, dict):
        raise ValueError(f"评测条目 {item_id} 的 tags 必须是字符串映射")
    for key, value in tags.items():
        key_ok = isinstance(key, str) and bool(key)
        value_ok = isinstance(value, str) and bool(value)
        if isinstance(value, list):
            value_ok = bool(value) and all(
                isinstance(member, str) and bool(member) for member in value
            )
        if not (key_ok and value_ok):
            raise ValueError(
                f"评测条目 {item_id} 的 tags 必须是非空字符串或字符串列表映射"
            )

def validate_fc_item(item: Any) -> dict[str, Any]:
    """function call 评测集单条校验, 返回原对象或抛 ValueError。"""
    if not isinstance(item, dict):
        raise ValueError("FC 条目必须是对象")
    if not isinstance(item.get("id"), str) or not item["id"]:
        raise ValueError("FC 条目缺少 id")
    if not isinstance(item.get("query"), str) or not item["query"]:
        raise ValueError(f"FC {item.get('id')} 缺少 query")
    tools = item.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"FC {item['id']} 缺少 tools 列表")
    for t in tools:
        fn = isinstance(t, dict) and t.get("function")
        if not (isinstance(fn, dict) and isinstance(fn.get("name"), str)):
            raise ValueError(f"FC {item['id']} 的 tools 结构非法 (需 function.name)")
    gold = item.get("gold")
    if not (isinstance(gold, dict) and isinstance(gold.get("name"), str)):
        raise ValueError(f"FC {item['id']} 缺少 gold.name")
    if "arguments" in gold and not isinstance(gold["arguments"], dict):
        raise ValueError(f"FC {item['id']} 的 gold.arguments 必须是对象")
    _validate_tags(item, item["id"])
    return item


def validate_subjective_item(item: Any) -> dict[str, Any]:
    """主观评测集单条校验。"""
    if not isinstance(item, dict):
        raise ValueError("主观条目必须是对象")
    if not isinstance(item.get("id"), str) or not item["id"]:
        raise ValueError("主观条目缺少 id")
    if not isinstance(item.get("query"), str) or not item["query"]:
        raise ValueError(f"主观 {item.get('id')} 缺少 query")
    if "reference" in item and not isinstance(item["reference"], str):
        raise ValueError(f"主观 {item['id']} 的 reference 必须是字符串")
    _validate_tags(item, item["id"])
    return item


def validate_evalset(records: list[Any], task_type: str) -> list[dict[str, Any]]:
    """整份评测集校验, 返回校验后的列表。"""
    if not isinstance(records, list) or not records:
        raise ValueError("评测集必须是非空数组")
    fn = validate_fc_item if task_type == "function_call" else validate_subjective_item
    return [fn(r) for r in records]
