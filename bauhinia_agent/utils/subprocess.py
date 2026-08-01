"""子进程执行通用工具。

shell、python_exec、diagnostics、grep 共用同一个 Popen 进程组边界，统一处理
超时/取消、进程树回收、TimeoutExpired / OSError 和输出截断。
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

from bauhinia_agent.runtime.cancellation import CancellationToken
from bauhinia_agent.utils.text import truncate


@dataclass(slots=True)
class CommandResult:
    """子进程执行的统一结果类型。

    工具层可以直接把 CommandResult 转成 ToolResult，
    不用每个工具重复处理 exit_code、stdout/stderr 截断等逻辑。
    """

    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    ok: bool
    error: str | None = None


def run_command(
    command: list[str] | str,
    *,
    cwd: Path,
    timeout_seconds: int = 30,
    max_output_chars: int = 20000,
    shell: bool = False,
    env: dict[str, str] | None = None,
    cancellation_token: CancellationToken | None = None,
) -> CommandResult:
    """执行子进程命令并返回统一结果。

    每个命令都在独立进程组中启动；超时或取消时终止整个进程组，并回收已经产生的输出。
    自动处理 TimeoutExpired 和 OSError，自动截断超长输出。
    这是 shell / python_exec / diagnostics / grep 四个工具共同需要的执行模式。
    """

    return _run_command_with_process_group(
        command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
        shell=shell,
        env=env,
        cancellation_token=cancellation_token,
    )


def _run_command_with_process_group(
    command: list[str] | str,
    *,
    cwd: Path,
    timeout_seconds: int,
    max_output_chars: int,
    shell: bool,
    env: dict[str, str] | None,
    cancellation_token: CancellationToken | None,
) -> CommandResult:
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            shell=shell,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_process_group_kwargs(),
        )
    except OSError as exc:
        return CommandResult(
            exit_code=-1,
            stdout="",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            ok=False,
            error=f"命令执行失败：{exc}",
        )
    _attach_windows_job(process)

    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    interrupted = False
    stdout = stderr = ""
    while True:
        if cancellation_token is not None and cancellation_token.is_cancelled:
            interrupted = True
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            stdout, stderr = process.communicate(
                timeout=min(remaining, 0.05) if cancellation_token is not None else remaining,
            )
            break
        except subprocess.TimeoutExpired:
            if cancellation_token is None:
                timed_out = True
                break

    if interrupted or timed_out:
        _terminate_process_group(process)
        # Drain output, but never wait forever for a descendant that inherited
        # stdout/stderr after the process-group termination request.
        stdout, stderr = _communicate_after_termination(process)
    stdout, stdout_truncated = truncate(stdout, max_output_chars)
    stderr, stderr_truncated = truncate(stderr, max_output_chars)
    _release_windows_job(process)

    if interrupted:
        return CommandResult(
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            ok=False,
            error="命令已中断",
        )
    if timed_out:
        return CommandResult(
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            ok=False,
            error="命令执行超时",
        )

    ok = process.returncode == 0
    return CommandResult(
        exit_code=process.returncode if process.returncode is not None else -1,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        ok=ok,
    )


def _process_group_kwargs() -> dict[str, int | bool]:
    """Start each command in its own process group/session."""

    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate the command and every descendant in its process group."""

    if os.name == "nt":
        if not _terminate_windows_job(process):
            _taskkill_process_tree(process.pid)
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    # The leader may have exited while a descendant ignored SIGTERM.  Kill the
    # group unconditionally after the grace period so those descendants cannot
    # survive merely because Popen's direct child is already reaped.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait(timeout=1)


def _taskkill_process_tree(pid: int) -> None:
    """Best-effort Windows equivalent of killing a POSIX process group."""

    try:
        taskkill = subprocess.Popen(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        try:
            taskkill.wait(timeout=0.75)
        except subprocess.TimeoutExpired:
            taskkill.kill()
            try:
                taskkill.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
    except OSError:
        pass


def _attach_windows_job(process: subprocess.Popen[str]) -> None:
    """Attach a Windows child tree to a killable Job Object when possible."""

    if os.name != "nt":
        return
    try:
        job = ctypes.windll.kernel32.CreateJobObjectW(None, None)
        if not job:
            return
        assigned = ctypes.windll.kernel32.AssignProcessToJobObject(job, process._handle)
        if not assigned:
            ctypes.windll.kernel32.CloseHandle(job)
            return
        process._bauhinia_job_handle = job
    except (AttributeError, OSError):
        return


def _terminate_windows_job(process: subprocess.Popen[str]) -> bool:
    if os.name != "nt":
        return False
    job = getattr(process, "_bauhinia_job_handle", None)
    if not job:
        return False
    try:
        return bool(ctypes.windll.kernel32.TerminateJobObject(job, 1))
    except (AttributeError, OSError):
        return False
    finally:
        try:
            ctypes.windll.kernel32.CloseHandle(job)
        except (AttributeError, OSError):
            pass
        process._bauhinia_job_handle = None


def _release_windows_job(process: subprocess.Popen[str]) -> None:
    if os.name != "nt":
        return
    job = getattr(process, "_bauhinia_job_handle", None)
    if not job:
        return
    try:
        ctypes.windll.kernel32.CloseHandle(job)
    except (AttributeError, OSError):
        pass
    process._bauhinia_job_handle = None


def _communicate_after_termination(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Collect terminated-process output without an unbounded pipe wait."""

    try:
        return process.communicate(timeout=1)
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_output(exc.stdout)
        stderr = _coerce_output(exc.stderr)
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        return stdout, stderr
    except (OSError, ValueError):
        return "", ""


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
