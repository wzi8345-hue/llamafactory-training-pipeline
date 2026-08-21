"""Narrow typed adapter for approved assistant actions and read-only observations."""

from __future__ import annotations

import hashlib

from . import datagen_job, dataset_store, eval_remote, eval_service, remote, train_service
from .assistant_schema import DataPlan, EvaluationPlan, TrainingPlan
from .eval_schema import EvalRequest


class AssistantTools:
    def __init__(self, cfg: remote.RemoteConfig | None = None):
        self.cfg = cfg

    def remote_config(self) -> remote.RemoteConfig:
        return self.cfg or remote.RemoteConfig.from_env()

    def new_job_id(self) -> str:
        return remote.new_job_id()

    def start_datagen(
        self,
        workflow_id: str,
        plan: DataPlan,
        launches: list[dict] | None = None,
    ) -> list[dict]:
        del workflow_id
        launches = launches or [
            {"job_id": self.new_job_id(), "task_type": item.task_type}
            for item in plan.items
        ]
        items = {item.task_type: item for item in plan.items}
        for launch in launches:
            item = items[launch["task_type"]]
            job_id = launch["job_id"]
            datagen_job.create_and_launch(job_id, item.config.model_dump())
        return launches

    def inspect_datagen(self, launches: list[dict]) -> list[dict]:
        observations = []
        for reference in launches:
            row = {**reference, **datagen_job.status(reference["job_id"])}
            if row.get("status") == "succeeded":
                row["output"] = str(
                    datagen_job.job_dir(reference["job_id"]) / "output.json"
                )
            observations.append(row)
        return observations

    def stop_external_job(self, kind: str, job_id: str) -> dict:
        """Route cancellation through the existing identity-safe stop primitive."""
        if kind == "datagen":
            result = datagen_job.stop(job_id)
            detail = str(result.get("detail") or "")
            terminal = bool(result.get("stopped")) or detail in {
                "无 pid",
                "进程身份已失效",
            }
        elif kind in {"training", "evaluation"}:
            result = remote.stop_job(self.remote_config(), job_id)
            detail = str(result.get("detail") or "")
            terminal = bool(result.get("stopped")) or detail in {
                "NOPROCESS",
                "STALEPID",
                "NOT_FOUND",
                "NOTFOUND",
            }
        else:
            raise ValueError(f"unsupported external job kind: {kind}")
        return {
            **result,
            "kind": kind,
            "job_id": job_id,
            "terminal": terminal,
        }

    def start_training(
        self, workflow_id: str, plan: TrainingPlan, job_id: str | None = None
    ) -> str:
        del workflow_id
        path = dataset_store.data_path(plan.dataset_name, "train")
        meta = dataset_store.dataset_meta(plan.dataset_name, "train")
        if meta is None or not path.exists():
            raise ValueError(f"dataset not found: {plan.dataset_name}")
        if plan.dataset_sha256:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != plan.dataset_sha256:
                raise ValueError(
                    f"dataset artifact hash changed: {plan.dataset_name}"
                )
        ref = train_service.TrainDataRef(
            path=path,
            ext=meta["ext"],
            source_type="assistant",
            source_id=plan.dataset_name,
        )
        return train_service.submit_training_job(
            self.remote_config(), plan.config, ref, plan.gpus, job_id=job_id
        )["job_id"]

    def inspect_training(self, job_id: str) -> dict:
        cfg = self.remote_config()
        status = remote.job_status(cfg, job_id)
        checkpoints_verified = True
        try:
            checkpoints = remote.list_checkpoints(cfg, job_id)
        except remote.RemoteError:
            checkpoints = []
            checkpoints_verified = False
        log_verified = True
        try:
            log_tail = remote.read_job_log_tail(cfg, job_id, lines=80)
        except remote.RemoteError:
            log_tail = ""
            log_verified = False
        log_state = (
            "unavailable"
            if not log_verified
            else ("empty" if not log_tail else "readable")
        )
        output_evidence = {
            "output_evidence_verified": False,
            "output_verified": False,
            "output_path": "",
            "output_state": "not_checked",
        }
        if status.get("status") == "succeeded":
            try:
                output_evidence = remote.inspect_training_output(cfg, job_id)
            except remote.RemoteError:
                output_evidence["output_state"] = "unavailable"
        metrics_verified = True
        try:
            metrics = remote.read_trainer_log(cfg, job_id)
        except (remote.RemoteError, ValueError, TypeError, AttributeError):
            metrics = {}
            metrics_verified = False
        gpus_verified = True
        try:
            gpus = remote.gpu_status(cfg)
        except (remote.RemoteError, ValueError, TypeError, AttributeError):
            gpus = []
            gpus_verified = False
        return {
            "job_id": job_id,
            "status": status,
            "metrics": metrics,
            "metrics_verified": metrics_verified,
            "gpus": gpus,
            "gpus_verified": gpus_verified,
            "checkpoints": checkpoints,
            "checkpoints_verified": checkpoints_verified,
            "log_tail": log_tail,
            "log_verified": log_verified,
            "log_state": log_state,
            **output_evidence,
        }

    def start_evaluation(
        self,
        workflow_id: str,
        req: EvalRequest,
        plan: EvaluationPlan,
        eval_id: str | None = None,
    ) -> str:
        del workflow_id
        for task_type, expected in plan.eval_sha256.items():
            name = plan.eval_dataset_names.get(task_type)
            if not name:
                raise ValueError(f"evaluation artifact is missing: {task_type}")
            path = dataset_store.data_path(name, "eval")
            if not path.exists():
                raise ValueError(f"evaluation artifact is missing: {name}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"evaluation artifact hash changed: {name}")
        return eval_service.submit_registered_eval(
            self.remote_config(),
            req,
            plan.eval_dataset_names.get("function_call"),
            plan.eval_dataset_names.get("subjective"),
            eval_id=eval_id,
        )

    def inspect_evaluation(self, eval_id: str) -> dict:
        cfg = self.remote_config()
        status = remote.job_status(cfg, eval_id)
        if status.get("status") not in {"failed", "interrupted"}:
            return status
        try:
            log_tail = remote.read_job_log_tail(
                cfg, eval_id, lines=80, log_name="eval.log"
            )
            return {
                **status,
                "log_tail": log_tail,
                "log_verified": True,
                "log_state": "empty" if not log_tail else "readable",
            }
        except remote.RemoteError:
            return {
                **status,
                "log_tail": "",
                "log_verified": False,
                "log_state": "unavailable",
            }

    def score_evaluation(
        self, eval_id: str, critical_tags: list[str] | tuple[str, ...] = ()
    ) -> dict:
        return eval_service.score_registered_eval(
            self.remote_config(), eval_id, critical_tags=critical_tags
        )

    def resolve_train_job(self, job_id: str) -> dict:
        return eval_remote.read_train_job(self.remote_config(), job_id)
