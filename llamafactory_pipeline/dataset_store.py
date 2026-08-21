"""本地数据集/评测集注册: 校验 → 落盘 → meta → 列表/删除。

数据存 sft_data/datasets/ (训练) 与 sft_data/evalsets/ (评测), 各带一份 .meta.json。
纯逻辑函数不依赖网络, 可单测; 路径由调用方注入 (便于测试)。
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from . import eval_schema, schema

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
Kind = Literal["train", "eval"]

_REPO = Path(__file__).resolve().parents[1]
_ROOT = _REPO / "sft_data"
_KIND_DIR = {"train": "datasets", "eval": "evalsets"}


def validate_name(name: str) -> str:
    if not name or not _NAME_RE.match(name):
        raise ValueError(f"数据集名非法 (需匹配 {_NAME_RE.pattern})")
    if name in (".", "..") or name.startswith("."):
        raise ValueError("数据集名不能以点开头")
    return name


def _dir(kind: Kind, base: Path | None = None) -> Path:
    root = Path(base) if base is not None else _ROOT
    return root / _KIND_DIR[kind]


def data_path(name: str, kind: Kind, base: Path | None = None) -> Path:
    """数据文件路径 (扩展名由注册时原文件决定, 存为 .json 或 .jsonl)。"""
    return _dir(kind, base) / f"{validate_name(name)}.data"


def meta_path(name: str, kind: Kind, base: Path | None = None) -> Path:
    return _dir(kind, base) / f"{validate_name(name)}.meta.json"


def _read_records(path: str, is_jsonl: bool) -> list[dict]:
    if is_jsonl:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    data = json.loads(Path(path).read_text("utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON 顶层必须是数组")
    return data


def register_dataset(
    src_path: str, name: str, kind: Kind, base: Path | None = None,
    source: str = "upload",
) -> dict[str, Any]:
    """校验 + 落盘数据 + 写 meta, 返回 meta。同名覆盖。

    - train: 复用 schema.read_first_record/detect_sharegpt_format 识别 ShareGPT 结构。
    - eval: 复用 eval_schema.validate_evalset 逐条校验 (FC/主观混合, 用 function_call 校验)。
    """
    validate_name(name)
    is_jsonl = src_path.endswith(".jsonl")
    records = _read_records(src_path, is_jsonl)
    if not records:
        raise ValueError("数据为空")

    meta: dict[str, Any] = {
        "name": name, "kind": kind, "n_records": len(records),
        "size_bytes": Path(src_path).stat().st_size,
        "source": source,
        "ext": ".jsonl" if is_jsonl else ".json",
    }

    d = _dir(kind, base)
    d.mkdir(parents=True, exist_ok=True)
    if kind == "train":
        fmt = schema.detect_sharegpt_format(records[0])
        meta["format"] = fmt
        meta["finetune_type"] = schema.detect_finetune_type(records[0])
    else:
        # 评测集: 尝试 FC 校验, 失败则当主观 (不强制结构, 打分阶段再判)
        try:
            eval_schema.validate_evalset(records, "function_call")
            meta["format"] = "function_call"
        except ValueError:
            meta["format"] = "subjective"

    # 落盘数据 (原样, 保留 jsonl 或 json)
    shutil.copyfile(src_path, data_path(name, kind, base))
    meta_path(name, kind, base).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def list_datasets(kind: Kind | None = None, base: Path | None = None) -> list[dict[str, Any]]:
    """列已注册数据集。kind=None 列两种。"""
    out = []
    kinds = [kind] if kind else ["train", "eval"]
    for k in kinds:
        d = _dir(k, base)
        if not d.exists():
            continue
        for p in sorted(d.glob("*.meta.json")):
            try:
                out.append(json.loads(p.read_text("utf-8")))
            except json.JSONDecodeError:
                continue
    return out


def dataset_meta(name: str, kind: Kind, base: Path | None = None) -> dict[str, Any] | None:
    p = meta_path(name, kind, base)
    if not p.exists():
        return None
    return json.loads(p.read_text("utf-8"))


def delete_dataset(name: str, kind: Kind, base: Path | None = None) -> bool:
    """删数据 + meta, 返回是否删了。"""
    validate_name(name)
    removed = False
    for p in (data_path(name, kind, base), meta_path(name, kind, base)):
        if p.exists():
            p.unlink()
            removed = True
    return removed
