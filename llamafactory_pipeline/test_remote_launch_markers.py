"""Offline fault-injection tests for durable remote launch markers."""

from __future__ import annotations

import os
import subprocess

import pytest

from . import eval_remote, remote


@pytest.mark.parametrize(
    ("markers", "expected"),
    [
        ({"status": "running"}, "ALREADY"),
        ({"pid": "999"}, "ALREADY"),
        ({"pid": "999", "pid_starttime": "1"}, "ALREADY"),
        ({"pid": "999", "pid_starttime": "1", "status": "running"}, "ALREADY"),
        ({"exit_code": "0"}, "ALREADY"),
    ],
)
@pytest.mark.parametrize("kind", ["training", "evaluation"])
def test_stale_partial_launch_markers_recover_with_the_same_job_id(
    tmp_path, monkeypatch, markers, expected, kind
):
    job_id = "20240101T000000Z-abcdef"
    cfg = remote.RemoteConfig("u@h", str(tmp_path), str(tmp_path))
    directory = tmp_path / job_id
    directory.mkdir()
    for name, value in markers.items():
        (directory / name).write_text(value, encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    flock = fake_bin / "flock"
    flock.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    flock.chmod(0o755)
    script = (
        remote.build_launch_script(cfg, job_id, "0")
        if kind == "training"
        else eval_remote.build_eval_launch(cfg, job_id, "0")
    )

    result = subprocess.run(
        ["bash"],
        input=script,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}"},
    )

    assert result.stdout.strip().splitlines()[-1] == expected
    if markers != {"exit_code": "0"}:
        assert (directory / "exit_code").read_text("utf-8") == "125"
        log_name = "train.log" if kind == "training" else "eval.log"
        assert "recovered incomplete" in (
            directory / log_name
        ).read_text("utf-8")


def test_status_only_crash_window_is_unknown_not_interrupted(tmp_path):
    job_id = "20240101T000000Z-abcdef"
    cfg = remote.RemoteConfig("u@h", str(tmp_path), str(tmp_path))
    directory = tmp_path / job_id
    directory.mkdir()
    (directory / "status").write_text("running", encoding="utf-8")

    result = subprocess.run(
        ["bash"],
        input=remote.build_status_script(cfg, job_id),
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "UNKNOWN"
