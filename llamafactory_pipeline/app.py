"""FastAPI 应用: 参数表单 + 数据上传 + 远端训练任务 + SSE 日志。

启动: python -m uvicorn llamafactory_pipeline.app:app --host 127.0.0.1 --port 8899
需先设置环境变量: TRAIN_SSH_TARGET / TRAIN_REMOTE_ROOT / LLAMAFACTORY_DIR
(可选 TRAIN_SSH_PORT / TRAIN_SSH_IDENTITY)。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import (datagen_job, datagen_schema, dataset_store, deploy, deploy_schema,
               eval_remote, eval_schema, eval_service, remote, schema)
from .train_service import (
    InsufficientRemoteDisk,
    TrainDataRef,
    submit_training_job,
)
from .assistant_api import router as assistant_router

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
_STATIC = Path(__file__).parent / "static"
_EVAL_RESULTS = Path(__file__).parent / "eval_results"

app = FastAPI(title="LlamaFactory 远程训练", version="0.1.0")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")
app.include_router(assistant_router)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/api/schema")
def get_schema() -> JSONResponse:
    return JSONResponse(schema.describe_schema())


@app.post("/api/jobs")
async def create_job(
    params: str = Form(...),
    file: Optional[UploadFile] = File(None),
    gpus: str = Form(""),
    from_datagen_job: str = Form(""),
    dataset_name: str = Form(""),
) -> JSONResponse:
    try:
        cfg = remote.RemoteConfig.from_env()
    except remote.RemoteError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 1. 校验训练参数
    try:
        train_cfg = schema.TrainConfig.model_validate(json.loads(params))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="params 不是合法 JSON")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"训练参数非法: {e}")

    # 2. 取训练数据: 上传文件 / 从生成任务导入 / 从数据集库选 (三选一)
    tmp_path, ext, is_tempfile = _resolve_train_data(file, from_datagen_job, dataset_name)
    source_type = "datagen" if from_datagen_job else "dataset" if dataset_name else "upload"
    source_id = from_datagen_job or dataset_name or (file.filename if file else "upload")
    try:
        try:
            result = submit_training_job(
                cfg,
                train_cfg,
                TrainDataRef(
                    path=Path(tmp_path),
                    ext=ext,
                    source_type=source_type,
                    source_id=source_id or "upload",
                    cleanup_after_submit=is_tempfile,
                ),
                gpus,
            )
        except InsufficientRemoteDisk as e:
            raise HTTPException(status_code=507, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except remote.RemoteError as e:
            raise HTTPException(status_code=502, detail=str(e))
    finally:
        if is_tempfile:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return JSONResponse(result)


def _resolve_train_data(
    file: Optional[UploadFile], from_datagen_job: str, dataset_name: str = ""
) -> tuple[str, str, bool]:
    """返回 (数据文件路径, 扩展名, 是否临时文件需清理)。

    - 上传文件: 校验扩展名 + 流式落临时文件 (is_tempfile=True)
    - from_datagen_job: 读生成任务的 output.json (is_tempfile=False, 直接用产物)
    - dataset_name: 读本地注册的训练数据集 (is_tempfile=False)
    三选一, 都没有则 400。
    """
    if from_datagen_job and dataset_name:
        raise HTTPException(status_code=400, detail="from_datagen_job 与 dataset_name 不能同时指定")
    if from_datagen_job:
        if file is not None:
            raise HTTPException(status_code=400, detail="不能同时上传文件和指定 from_datagen_job")
        try:
            remote.validate_job_id(from_datagen_job)
        except remote.RemoteError:
            raise HTTPException(status_code=400, detail="非法的生成任务 ID")
        from . import datagen_job
        out = datagen_job.job_dir(from_datagen_job) / "output.json"
        if not out.exists():
            raise HTTPException(status_code=404, detail=f"生成任务 {from_datagen_job} 无产出 output.json")
        return str(out), ".json", False

    if dataset_name:
        if file is not None:
            raise HTTPException(status_code=400, detail="不能同时上传文件和指定 dataset_name")
        try:
            p = dataset_store.data_path(dataset_name, "train")
            m = dataset_store.dataset_meta(dataset_name, "train")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not p.exists() or m is None:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_name} 不存在")
        return str(p), m.get("ext", ".json"), False

    if file is None:
        raise HTTPException(status_code=400, detail="需上传文件或指定 from_datagen_job / dataset_name")

    filename = os.path.basename(file.filename or "")
    ext = Path(filename).suffix.lower()
    if ext not in (".json", ".jsonl"):
        raise HTTPException(status_code=400, detail="只接受 .json 或 .jsonl")
    return _save_upload_capped(file), ext, True


@app.get("/api/jobs")
def list_jobs() -> JSONResponse:
    """所有历史任务 (训练+评测) 及状态, 最新在前。"""
    try:
        cfg = remote.RemoteConfig.from_env()
        return JSONResponse({"jobs": remote.list_jobs(cfg)})
    except remote.RemoteError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> JSONResponse:
    """停止运行中的任务 (train/eval 通用)。"""
    try:
        cfg = remote.RemoteConfig.from_env()
        remote.validate_job_id(job_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        return JSONResponse(remote.stop_job(cfg, job_id))
    except remote.RemoteError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/disk")
def disk_status() -> JSONResponse:
    """remote_root 所在盘可用空间 (字节)。"""
    try:
        cfg = remote.RemoteConfig.from_env()
        return JSONResponse({"avail_bytes": remote.disk_free(cfg)})
    except remote.RemoteError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    try:
        cfg = remote.RemoteConfig.from_env()
        remote.validate_job_id(job_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(remote.job_status(cfg, job_id))


@app.get("/api/jobs/{job_id}/logs")
def get_logs(job_id: str, request: Request) -> StreamingResponse:
    try:
        cfg = remote.RemoteConfig.from_env()
        remote.validate_job_id(job_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))

    last_id = request.headers.get("Last-Event-ID")
    from_line = (int(last_id) + 1) if (last_id and last_id.isdigit()) else 1

    def gen():
        for line_no, content in remote.stream_logs(cfg, job_id, from_line):
            if content == "\0heartbeat":
                yield ": keepalive\n\n"
                continue
            yield f"id: {line_no}\ndata: {content}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/metrics")
def job_metrics(job_id: str) -> JSONResponse:
    """训练指标 (loss 曲线数据), 前端轮询实时绘制。"""
    try:
        cfg = remote.RemoteConfig.from_env()
        remote.validate_job_id(job_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        return JSONResponse(remote.read_trainer_log(cfg, job_id))
    except remote.RemoteError:
        return JSONResponse({"points": [], "total_steps": 0})


@app.get("/api/jobs/{job_id}/checkpoints")
def job_checkpoints(job_id: str) -> JSONResponse:
    """列某训练任务的 checkpoint-* 子目录, 供续训选择 (返回绝对路径)。"""
    try:
        cfg = remote.RemoteConfig.from_env()
        remote.validate_job_id(job_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        ckpts = remote.list_checkpoints(cfg, job_id)
        out_dir = remote.resolve_output_dir(cfg, job_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=502, detail=str(e))
    # 拼绝对路径: 前端续训直接填 resume_from_checkpoint
    for c in ckpts:
        c["path"] = f"{out_dir.rstrip('/')}/{c['name']}"
    return JSONResponse({"output_dir": out_dir, "checkpoints": ckpts})


@app.post("/api/jobs/{job_id}/cleanup")
def cleanup_checkpoint(job_id: str, checkpoint_name: str = Form(...)) -> JSONResponse:
    """删除某训练任务的指定 checkpoint-* (防误删: 名必须形如 checkpoint-N)。"""
    try:
        cfg = remote.RemoteConfig.from_env()
        remote.validate_job_id(job_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        return JSONResponse(remote.cleanup_checkpoint(cfg, job_id, checkpoint_name))
    except remote.RemoteError as e:
        # 非法 checkpoint 名 (含注入) → 400; 远端失败 → 502
        code = 400 if "必须形如" in str(e) else 502
        raise HTTPException(status_code=code, detail=str(e))


@app.get("/api/jobs/{job_id}/download")
def download_checkpoint(job_id: str, name: str) -> StreamingResponse:
    """流式下载某训练产物 (checkpoint-* 或整个 output_dir), 远端 tar 流回。"""
    try:
        cfg = remote.RemoteConfig.from_env()
        remote.validate_job_id(job_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # name 形如 checkpoint-N, 校验防 traversal
    import re as _re
    if not _re.match(r"^checkpoint-\d+$", name):
        raise HTTPException(status_code=400, detail="name 必须形如 checkpoint-<数字>")
    try:
        out_dir = remote.resolve_output_dir(cfg, job_id)
        if not out_dir:
            raise HTTPException(status_code=404, detail="该任务无 output_dir")
    except remote.RemoteError as e:
        raise HTTPException(status_code=502, detail=str(e))
    target = f"{out_dir.rstrip('/')}/{name}"
    # 远端 tar 流式输出, 经 SSH stdout 拉回
    argv = remote._ssh_argv(cfg) + [f"tar -cf - -C {out_dir} {name}"]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return StreamingResponse(proc.stdout, media_type="application/x-tar",
                             headers={"Content-Disposition": f'attachment; filename="{name}.tar"'})


@app.get("/api/gpus")
def gpu_status() -> JSONResponse:
    """服务器实时显卡状态 (训练前/中均可轮询)。"""
    try:
        cfg = remote.RemoteConfig.from_env()
        return JSONResponse({"gpus": remote.gpu_status(cfg)})
    except remote.RemoteError as e:
        raise HTTPException(status_code=502, detail=str(e))


def _save_upload_capped(file: UploadFile) -> str:
    """流式写临时文件, 超过上限即中止并删除。"""
    fd, path = tempfile.mkstemp(suffix=".upload")
    total = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="文件超过 500MB")
                out.write(chunk)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


# ─────────────────────────── 评测子系统 ───────────────────────────

@app.get("/api/eval/schema")
def eval_schema_info() -> JSONResponse:
    return JSONResponse({
        "default_fc_param_prompt": eval_schema.DEFAULT_FC_PARAM_PROMPT,
        "default_subjective_prompt": eval_schema.DEFAULT_SUBJECTIVE_PROMPT,
        "task_types": ["function_call", "subjective"],
    })


def _read_records(path: str, ext: str) -> list:
    """读上传的评测集 (json 数组或 jsonl)。"""
    txt = Path(path).read_text(encoding="utf-8").strip()
    if ext == ".jsonl":
        return [json.loads(l) for l in txt.splitlines() if l.strip()]
    data = json.loads(txt)
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="JSON 评测集顶层必须是数组")
    return data


def _save_evalset_jsonl(records: list, results_dir: Path, name: str) -> str:
    """把校验后的评测集规范化为 jsonl 存到结果目录, 返回路径 (供 scp 与打分复用)。"""
    p = results_dir / f"{name}.items.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(p)


@app.post("/api/eval/jobs")
async def create_eval_job(
    config: str = Form(...),
    fc_file: Optional[UploadFile] = File(None),
    subjective_file: Optional[UploadFile] = File(None),
) -> JSONResponse:
    try:
        cfg = remote.RemoteConfig.from_env()
    except remote.RemoteError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        raw = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="config 不是合法 JSON")

    # 微调模型: 由已完成训练任务解析 base/adapter/template
    models = list(raw.get("models", []))
    for fj in raw.get("finetuned_jobs", []):
        try:
            resolved = eval_remote.read_train_job(cfg, fj["job_id"])
        except (remote.RemoteError, KeyError) as e:
            raise HTTPException(status_code=400, detail=f"解析训练任务失败: {e}")
        if not resolved.get("model_name_or_path"):
            raise HTTPException(status_code=400, detail=f"任务 {fj.get('job_id')} 无有效 base 模型")
        models.append({
            "name": fj["name"],
            "model_name_or_path": resolved["model_name_or_path"],
            "adapter_path": resolved["adapter_path"] or None,
            "template": resolved["template"],
        })

    # 从干净字段重建, 避免 finetuned_jobs 这类非模型字段触发 extra 报错
    try:
        req = eval_schema.EvalRequest(
            models=models,
            task_types=raw["task_types"],
            gpus=raw.get("gpus", ""),
            api_port=raw.get("api_port", 8000),
            ready_timeout=raw.get("ready_timeout", 600),
            fc_param_prompt=raw.get("fc_param_prompt", eval_schema.DEFAULT_FC_PARAM_PROMPT),
            subjective_prompt=raw.get("subjective_prompt", eval_schema.DEFAULT_SUBJECTIVE_PROMPT),
        )
        req.validate_names()
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"缺少字段: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"评测配置非法: {e}")

    need_fc = "function_call" in req.task_types
    need_subj = "subjective" in req.task_types
    fc_ds = raw.get("fc_dataset_name", "")
    subj_ds = raw.get("subj_dataset_name", "")
    if need_fc and fc_file is None and not fc_ds:
        raise HTTPException(status_code=400, detail="选择了 function_call 但未上传 FC 评测集或指定 fc_dataset_name")
    if need_subj and subjective_file is None and not subj_ds:
        raise HTTPException(status_code=400, detail="选择了 subjective 但未上传主观评测集或指定 subj_dataset_name")

    eval_id = eval_remote.new_eval_id()
    results_dir = _EVAL_RESULTS / eval_id
    results_dir.mkdir(parents=True, exist_ok=True)

    fc_local = subj_local = None
    if need_fc:
        fc_local = _resolve_evalset(fc_file, fc_ds, "function_call", results_dir, "fc")
    if need_subj:
        subj_local = _resolve_evalset(subjective_file, subj_ds, "subjective", results_dir, "subjective")

    try:
        eval_service.submit_normalized_eval(
            cfg, req, eval_id, results_dir, fc_local, subj_local
        )
    except remote.RemoteError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return JSONResponse({"eval_id": eval_id, "models": [m.name for m in req.models]})


def _prepare_evalset(
    file: UploadFile, task_type: str, results_dir: Path, name: str
) -> tuple[str, list]:
    """保存上传 → 校验 → 规范化为 jsonl。返回 (jsonl路径, records)。"""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".json", ".jsonl"):
        raise HTTPException(status_code=400, detail=f"{name} 评测集只接受 .json/.jsonl")
    up = _save_upload_capped(file)
    try:
        records = _read_records(up, ext)
        try:
            records = eval_schema.validate_evalset(records, task_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{name} 评测集校验失败: {e}")
    finally:
        try:
            os.unlink(up)
        except OSError:
            pass
    return _save_evalset_jsonl(records, results_dir, name), records


def _resolve_evalset(
    file: Optional[UploadFile], dataset_name: str, task_type: str,
    results_dir: Path, name: str,
) -> str:
    """评测集来源: 上传文件 或 本地注册数据集。返回规范化 jsonl 路径。"""
    if dataset_name:
        try:
            p = dataset_store.data_path(dataset_name, "eval")
            m = dataset_store.dataset_meta(dataset_name, "eval")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not p.exists() or m is None:
            raise HTTPException(status_code=404, detail=f"评测集 {dataset_name} 不存在")
        records = _read_records(str(p), m.get("ext", ".json") == ".jsonl")
        try:
            records = eval_schema.validate_evalset(records, task_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{name} 评测集校验失败: {e}")
        return _save_evalset_jsonl(records, results_dir, name)
    # 上传路径
    assert file is not None  # 调用方已校验非空
    return _prepare_evalset(file, task_type, results_dir, name)[0]


@app.get("/api/eval/jobs/{eval_id}")
def get_eval_job(eval_id: str) -> JSONResponse:
    try:
        cfg = remote.RemoteConfig.from_env()
        remote.validate_job_id(eval_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(remote.job_status(cfg, eval_id))


@app.get("/api/eval/jobs/{eval_id}/logs")
def get_eval_logs(eval_id: str, request: Request) -> StreamingResponse:
    try:
        cfg = remote.RemoteConfig.from_env()
        remote.validate_job_id(eval_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))

    last_id = request.headers.get("Last-Event-ID")
    from_line = (int(last_id) + 1) if (last_id and last_id.isdigit()) else 1

    def gen():
        for line_no, content in remote.stream_logs(cfg, eval_id, from_line, log_name="eval.log"):
            if content == "\0heartbeat":
                yield ": keepalive\n\n"
                continue
            yield f"id: {line_no}\ndata: {content}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/eval/jobs/{eval_id}/score")
def score_eval_job(eval_id: str) -> JSONResponse:
    try:
        cfg = remote.RemoteConfig.from_env()
        remote.validate_job_id(eval_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not (_EVAL_RESULTS / eval_id / "config.json").exists():
        raise HTTPException(status_code=404, detail="找不到该评测的配置, 无法打分")
    try:
        summary = eval_service.score_registered_eval(
            cfg, eval_id, results_root=_EVAL_RESULTS
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"打分失败: {e}")
    return JSONResponse(summary)


# ─────────────────────────── 数据生成子系统 ───────────────────────────

@app.get("/api/datagen/schema")
def datagen_schema_info() -> JSONResponse:
    return JSONResponse({
        "default_qa_gen_prompt": datagen_schema.DEFAULT_QA_GEN_PROMPT,
        "default_qa_judge_prompt": datagen_schema.DEFAULT_QA_JUDGE_PROMPT,
        "default_fc_gen_prompt": datagen_schema.DEFAULT_FC_GEN_PROMPT,
        "default_fc_judge_prompt": datagen_schema.DEFAULT_FC_JUDGE_PROMPT,
        "default_qa_multi_gen_synthesis": datagen_schema.DEFAULT_QA_MULTI_GEN_SYNTHESIS,
        "default_qa_multi_gen_consensus": datagen_schema.DEFAULT_QA_MULTI_GEN_CONSENSUS,
        "default_qa_multi_judge_prompt": datagen_schema.DEFAULT_QA_MULTI_JUDGE_PROMPT,
        "default_qa_dpo_rejected_prompt": datagen_schema.DEFAULT_QA_DPO_REJECTED_PROMPT,
        "default_fc_dpo_rejected_prompt": datagen_schema.DEFAULT_FC_DPO_REJECTED_PROMPT,
        "default_dpo_pair_judge_prompt": datagen_schema.DEFAULT_DPO_PAIR_JUDGE_PROMPT,
        "finetune_types": ["sft", "dpo"],
        "task_types": ["qa", "fc", "qa_multi"],
    })


@app.get("/api/datagen/jobs")
def list_datagen_jobs() -> JSONResponse:
    """所有数据生成任务及状态, 最新在前 (供"从生成任务导入"下拉用)。"""
    return JSONResponse({"jobs": datagen_job.list_jobs()})


@app.post("/api/datagen/jobs")
def create_datagen_job(config: str = Form(...)) -> JSONResponse:
    try:
        raw = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="config 不是合法 JSON")
    try:
        cfg = datagen_schema.DatagenConfig.model_validate(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"生成配置非法: {e}")

    job_id = remote.new_job_id()
    try:
        datagen_job.create_and_launch(job_id, cfg.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动生成失败: {e}")
    return JSONResponse({"job_id": job_id, "finetune_type": cfg.finetune_type,
                         "task_type": cfg.task_type, "target": cfg.count})


@app.get("/api/datagen/jobs/{job_id}")
def get_datagen_job(job_id: str) -> JSONResponse:
    try:
        remote.validate_job_id(job_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(datagen_job.status(job_id))


@app.post("/api/datagen/jobs/{job_id}/stop")
def stop_datagen_job(job_id: str) -> JSONResponse:
    try:
        remote.validate_job_id(job_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(datagen_job.stop(job_id))


@app.get("/api/datagen/jobs/{job_id}/logs")
def get_datagen_logs(job_id: str, request: Request) -> StreamingResponse:
    try:
        remote.validate_job_id(job_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))
    last_id = request.headers.get("Last-Event-ID")
    from_line = (int(last_id) + 1) if (last_id and last_id.isdigit()) else 1

    def gen():
        for line_no, content in datagen_job.tail_logs(job_id, from_line):
            if content == "\0heartbeat":
                yield ": keepalive\n\n"
                continue
            yield f"id: {line_no}\ndata: {content}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/datagen/jobs/{job_id}/report")
def get_datagen_report(job_id: str) -> JSONResponse:
    try:
        remote.validate_job_id(job_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))
    report = datagen_job.job_dir(job_id) / "report.md"
    if not report.exists():
        raise HTTPException(status_code=404, detail="报告尚未生成")
    return JSONResponse({"report_md": report.read_text(encoding="utf-8")})


@app.get("/api/datagen/jobs/{job_id}/download")
def download_datagen_output(job_id: str) -> FileResponse:
    try:
        remote.validate_job_id(job_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))
    out = datagen_job.job_dir(job_id) / "output.json"
    if not out.exists():
        raise HTTPException(status_code=404, detail="产出尚未生成")
    finetune_type = datagen_job.job_finetune_type(job_id)
    return FileResponse(
        out, filename=f"{finetune_type}_{job_id}.json", media_type="application/json")


@app.get("/api/eval/jobs/{eval_id}/report")
def get_eval_report(eval_id: str) -> JSONResponse:
    try:
        remote.validate_job_id(eval_id)
    except remote.RemoteError as e:
        raise HTTPException(status_code=400, detail=str(e))
    report = _EVAL_RESULTS / eval_id / "report.md"
    if not report.exists():
        raise HTTPException(status_code=404, detail="报告尚未生成, 请先打分")
    return JSONResponse({"report_md": report.read_text(encoding="utf-8")})


# ─────────────────────────── 模型部署子系统 ───────────────────────────

@app.get("/api/deploy/schema")
def deploy_schema_info() -> JSONResponse:
    return JSONResponse(deploy_schema.describe_schema())


@app.post("/api/deploy")
def create_deployment(config: str = Form(...)) -> JSONResponse:
    try:
        raw = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="config 不是合法 JSON")
    try:
        d = deploy_schema.DeployConfig.model_validate(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"部署配置非法: {e}")
    try:
        d = d.normalized()  # 容器名校验 (非法字符/前缀)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        cfg = remote.RemoteConfig.from_env()
    except remote.RemoteError as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        return JSONResponse(deploy.deploy(cfg, d))
    except remote.RemoteError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/deploy")
def list_deployments() -> JSONResponse:
    try:
        cfg = remote.RemoteConfig.from_env()
    except remote.RemoteError as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        containers = deploy.list_deployments(cfg)
    except remote.RemoteError as e:
        raise HTTPException(status_code=502, detail=str(e))
    # 合并本地配置 (有配置但容器已删的也列出, 标 missing)
    by_name = {c["name"]: c for c in containers if c.get("name")}
    for name in deploy.list_saved_configs():
        if name not in by_name:
            by_name[name] = {"name": name, "status": "missing", "status_text": "", "ports": ""}
    return JSONResponse({"deployments": list(by_name.values())})


@app.get("/api/deploy/{name}")
def deployment_status(name: str) -> JSONResponse:
    try:
        cfg = remote.RemoteConfig.from_env()
    except remote.RemoteError as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        return JSONResponse(deploy.deployment_status(cfg, name))
    except remote.RemoteError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/deploy/{name}/stop")
def stop_deployment(name: str) -> JSONResponse:
    try:
        cfg = remote.RemoteConfig.from_env()
    except remote.RemoteError as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        return JSONResponse(deploy.stop_deployment(cfg, name))
    except remote.RemoteError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.delete("/api/deploy/{name}")
def delete_deployment(name: str) -> JSONResponse:
    """停止+删除容器+删本地配置。"""
    try:
        cfg = remote.RemoteConfig.from_env()
    except remote.RemoteError as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        deploy.stop_deployment(cfg, name)
    except remote.RemoteError as e:
        raise HTTPException(status_code=502, detail=str(e))
    deploy.delete_config(name)
    return JSONResponse({"deleted": True, "name": name})


@app.get("/api/deploy/{name}/logs")
def deploy_logs(name: str, request: Request) -> StreamingResponse:
    try:
        cfg = remote.RemoteConfig.from_env()
    except remote.RemoteError as e:
        raise HTTPException(status_code=500, detail=str(e))
    last_id = request.headers.get("Last-Event-ID")
    from_line = (int(last_id) + 1) if (last_id and last_id.isdigit()) else 1

    def gen():
        for line_no, content in deploy.stream_deploy_logs(cfg, name, from_line):
            if content == "\0heartbeat":
                yield ": keepalive\n\n"
                continue
            yield f"id: {line_no}\ndata: {content}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/deploy/{name}/health")
def deploy_health(name: str) -> JSONResponse:
    """探 vLLM /health 就绪态。"""
    try:
        cfg = remote.RemoteConfig.from_env()
    except remote.RemoteError as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        return JSONResponse(deploy.probe_health(cfg, name))
    except deploy.DeployError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except remote.RemoteError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/deploy/{name}/metrics")
def deploy_metrics(name: str) -> JSONResponse:
    """抓 vLLM /metrics 并解析。失败返回空 (监控页容错)。"""
    try:
        cfg = remote.RemoteConfig.from_env()
    except remote.RemoteError as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        return JSONResponse(deploy.fetch_metrics(cfg, name))
    except deploy.DeployError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/deploy/{name}/chat")
async def deploy_chat(name: str, request: Request) -> StreamingResponse:
    """流式代理 chat completions 到部署的 vLLM 端点。"""
    try:
        cfg = remote.RemoteConfig.from_env()
    except remote.RemoteError as e:
        raise HTTPException(status_code=500, detail=str(e))
    body = await request.body()
    # 先同步解析端点 (无配置立即 400, 不进生成器延迟抛)
    try:
        deploy._resolve_endpoint(cfg, name)
    except deploy.DeployError as e:
        raise HTTPException(status_code=400, detail=str(e))
    gen = deploy.chat_proxy_stream(cfg, name, body)
    return StreamingResponse(gen, media_type="text/event-stream")


# ─────────────────────────── 数据管理子系统 ───────────────────────────

@app.get("/api/datasets")
def list_datasets(kind: Optional[str] = None) -> JSONResponse:
    """列已注册数据集/评测集。kind=train|eval|None。"""
    try:
        items = dataset_store.list_datasets(kind=kind)  # type: ignore[arg-type]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"items": items})


@app.post("/api/datasets")
async def register_dataset(
    name: str = Form(...),
    kind: str = Form(...),
    file: UploadFile = File(...),
) -> JSONResponse:
    if kind not in ("train", "eval"):
        raise HTTPException(status_code=400, detail="kind 需为 train 或 eval")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".json", ".jsonl"):
        raise HTTPException(status_code=400, detail="只接受 .json 或 .jsonl")
    tmp = _save_upload_capped(file)
    try:
        meta = dataset_store.register_dataset(tmp, name, kind, source="upload")  # type: ignore[arg-type]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return JSONResponse(meta)


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str, kind: str = "train") -> JSONResponse:
    try:
        removed = dataset_store.delete_dataset(name, kind)  # type: ignore[arg-type]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"deleted": removed, "name": name})
