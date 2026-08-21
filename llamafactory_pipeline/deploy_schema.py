"""vLLM docker 部署参数模型 + 表单描述。

字段对应 vLLM/vllm-openai 镜像启动参数, 纯逻辑, 不做网络/SSH。
参考启动命令见 docs/部署示例。
"""

from __future__ import annotations

import re
from typing import Any, Optional, get_args, get_origin, Literal

from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid")
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_PREFIX = "vllm-"


def normalize_container_name(name: str) -> str:
    """强制 vllm- 前缀 + 安全字符校验, 防注入与命名冲突。"""
    n = (name or "").strip()
    if not n:
        raise ValueError("容器名不能为空")
    if not n.startswith(_PREFIX):
        n = _PREFIX + n
    body = n[len(_PREFIX):]
    if not body or not _NAME_RE.match(body):
        raise ValueError(f"容器名非法 (去前缀后需匹配 {_NAME_RE.pattern})")
    return n


class DeployConfig(BaseModel):
    """vLLM docker 部署配置。container_name 会自动补 vllm- 前缀。"""
    model_config = _STRICT

    container_name: str                            # 自动补 vllm- 前缀
    image: str = "vllm/vllm-openai:latest"
    host_model_path: str                           # 宿主机权重路径, -v 挂载源
    model_path: str = ""                           # 容器内路径; 空则从 host_model_path basename 推导
    gpus: str = ""                                 # ""=all, "3"=device=3, "0,1"=device=0,1
    port: int = 8000
    api_key: str = ""                              # 空=不传 --api-key (无鉴权)
    restart_policy: str = "unless-stopped"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 128
    reasoning_parser: str = ""                     # 空=不传
    enable_auto_tool_choice: bool = False
    tool_call_parser: str = ""                     # 空=不传
    speculative_config: str = ""                   # JSON 字符串原样透传
    enable_lora: bool = False
    lora_modules: str = ""                         # 如 "name1=/path1 name2=/path2"
    max_lora_rank: int = 0                         # 0=不传
    extra_args: str = ""                           # 透传未建模参数 (原样追加)

    def normalized(self) -> "DeployConfig":
        """返回 container_name 已规范化的副本 (补前缀)。"""
        return self.model_copy(update={"container_name": normalize_container_name(self.container_name)})

    def resolved_model_path(self) -> str:
        """容器内模型路径: 未填则用 host_model_path 的 basename 挂到 /models 下。"""
        if self.model_path.strip():
            return self.model_path.strip()
        import os
        return "/models/" + os.path.basename(self.host_model_path.rstrip("/"))


# ── 表单描述 (复用 schema.py 的 _field_type 思路) ──

def _field_type(annotation: Any) -> tuple[str, Optional[list[str]]]:
    origin = get_origin(annotation)
    import typing
    if origin is typing.Union:
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if inner:
            return _field_type(inner[0])
    if origin is Literal:
        return "select", [str(v) for v in get_args(annotation)]
    if annotation is bool:
        return "bool", None
    if annotation is int:
        return "int", None
    if annotation is float:
        return "float", None
    return "str", None


def describe_schema() -> dict[str, Any]:
    """生成扁平字段列表供前端渲染 (字段少, 不分组)。"""
    fields = []
    for fname, finfo in DeployConfig.model_fields.items():
        ftype, enum = _field_type(finfo.annotation)
        default = finfo.get_default(call_default_factory=True)
        # 必填字段无默认值 (PydanticUndefined), JSON 序列化用 None
        if finfo.is_required():
            default = None
        fields.append({
            "name": fname,
            "type": ftype,
            "enum": enum,
            "default": default,
            "required": finfo.is_required(),
        })
    return {"fields": fields, "name_prefix": _PREFIX}
