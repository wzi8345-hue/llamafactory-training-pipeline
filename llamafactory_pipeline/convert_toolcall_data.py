"""把 OpenAI 风格 tool_calls 的 ShareGPT 数据转成 LlamaFactory 原生工具格式。

输入每条: conversations(from/value, gpt turn 带 tool_calls) + tools(list)。
输出每条: gpt 的工具调用 → from="function_call", value=JSON 字符串; tools → JSON 字符串。
其余 human/gpt 文本轮原样保留。LlamaFactory 默认 sharegpt tags 即可读 (function_tag=function_call)。

用法: python -m llamafactory_pipeline.convert_toolcall_data 输入.json 输出.json
自检: python -m llamafactory_pipeline.convert_toolcall_data --selftest
"""

from __future__ import annotations

import json
import sys


def convert_record(r: dict) -> dict:
    tools = r.get("tools")
    out: dict = {"conversations": []}
    for m in r.get("conversations", []):
        if m.get("from") == "gpt" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                args = tc.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass  # 解析失败保留原字符串, 不丢数据
                calls.append({"name": tc.get("name"), "arguments": args})
            value = calls[0] if len(calls) == 1 else calls
            out["conversations"].append(
                {"from": "function_call", "value": json.dumps(value, ensure_ascii=False)})
        else:
            out["conversations"].append({"from": m.get("from"), "value": m.get("value", "")})
    if isinstance(tools, (list, dict)):
        out["tools"] = json.dumps(tools, ensure_ascii=False)
    elif tools is not None:
        out["tools"] = tools
    if "system" in r:
        out["system"] = r["system"]
    return out


def convert_file(src: str, dst: str) -> int:
    data = json.load(open(src, encoding="utf-8"))
    conv = [convert_record(r) for r in data]
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)
    return len(conv)


def _selftest() -> None:
    rec = {
        "conversations": [
            {"from": "human", "value": "分析下"},
            {"from": "gpt", "value": "请补充信息"},
            {"from": "human", "value": "钢的腐蚀"},
            {"from": "gpt", "value": "", "tool_calls": [
                {"name": "plan", "arguments": "{\"kw\": [\"腐蚀\"]}"}]},
        ],
        "tools": [{"type": "function", "function": {"name": "plan"}}],
    }
    out = convert_record(rec)
    conv = out["conversations"]
    assert conv[0] == {"from": "human", "value": "分析下"}
    assert conv[1] == {"from": "gpt", "value": "请补充信息"}
    assert conv[3]["from"] == "function_call"
    assert json.loads(conv[3]["value"]) == {"name": "plan", "arguments": {"kw": ["腐蚀"]}}
    assert isinstance(out["tools"], str) and json.loads(out["tools"])[0]["function"]["name"] == "plan"
    print("selftest OK")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        _selftest()
    elif len(sys.argv) == 3:
        n = convert_file(sys.argv[1], sys.argv[2])
        print(f"已转换 {n} 条 → {sys.argv[2]}")
    else:
        print(__doc__)
        sys.exit(1)
