"""Minimal HTTP sandbox: executes Python in a subprocess with strict limits."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import uuid
from contextlib import redirect_stderr, redirect_stdout

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="LLM Research Sandbox", version="1.0.0")

EXEC_TIMEOUT = int(os.environ.get("SANDBOX_EXEC_TIMEOUT_SEC", "45"))
MAX_OUT = int(os.environ.get("SANDBOX_MAX_OUTPUT_BYTES", str(256 * 1024)))


class ExecuteRequest(BaseModel):
    code: str = Field(..., description="Python source to execute in isolated subprocess")
    timeout_sec: int | None = Field(default=None, ge=1, le=120)


class ExecuteResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    execution_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/execute", response_model=ExecuteResponse)
def execute(req: ExecuteRequest) -> ExecuteResponse:
    if not req.code or not req.code.strip():
        raise HTTPException(status_code=400, detail="empty code")
    timeout = min(req.timeout_sec or EXEC_TIMEOUT, 120)
    eid = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        dir="/tmp",
        encoding="utf-8",
    ) as f:
        f.write(req.code)
        path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONHASHSEED": "0",
                "SANDBOX_EXECUTION_ID": eid,
            },
        )
        out = (proc.stdout or "")[:MAX_OUT]
        err = (proc.stderr or "")[:MAX_OUT]
        return ExecuteResponse(
            stdout=out,
            stderr=err,
            exit_code=proc.returncode,
            execution_id=eid,
        )
    except subprocess.TimeoutExpired:
        return ExecuteResponse(
            stdout="",
            stderr=f"timeout after {timeout}s",
            exit_code=-9,
            execution_id=eid,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/execute_inline")
def execute_inline(req: ExecuteRequest) -> ExecuteResponse:
    """Run code in-process (same UID); stricter resource limits should use /execute."""
    timeout = min(req.timeout_sec or EXEC_TIMEOUT, 60)
    eid = str(uuid.uuid4())
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    g: dict = {"__name__": "__sandbox__"}
    exit_code = 0
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            exec(compile(req.code, "<sandbox>", "exec"), g, g)
    except Exception as e:
        exit_code = 1
        print(repr(e), file=buf_err)
    return ExecuteResponse(
        stdout=buf_out.getvalue()[:MAX_OUT],
        stderr=buf_err.getvalue()[:MAX_OUT],
        exit_code=exit_code,
        execution_id=eid,
    )
