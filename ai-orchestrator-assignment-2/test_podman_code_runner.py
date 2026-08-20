"""Pytest tests for the Podman code runner with a mocked container API."""

from __future__ import annotations

import base64
import importlib
import sys
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fake container infrastructure (no real Podman required)
# ---------------------------------------------------------------------------

class _FakeContainer:
    def __init__(
        self,
        wait_result: dict | int = {"StatusCode": 0},
        logs_result: bytes = b"3628800\n",
        raise_on_wait: bool = False,
    ) -> None:
        self.wait_result = wait_result
        self.logs_result = logs_result
        self.raise_on_wait = raise_on_wait
        self.removed = False
        self.killed = False

    def wait(self, timeout: int) -> dict | int:
        if self.raise_on_wait:
            raise TimeoutError("container timed out")
        return self.wait_result

    def logs(self, stdout: bool = True, stderr: bool = True) -> bytes:
        return self.logs_result

    def kill(self) -> None:
        self.killed = True

    def remove(self, force: bool = False) -> None:
        self.removed = True


class _FakeContainers:
    def __init__(self, container: _FakeContainer) -> None:
        self._container = container
        self.run_kwargs: dict | None = None

    def run(self, **kwargs) -> _FakeContainer:
        self.run_kwargs = kwargs
        return self._container


class _FakeImages:
    def __init__(self) -> None:
        self.build_calls: list[dict] = []

    def build(self, **kwargs) -> None:
        self.build_calls.append(kwargs)


class _FakeClient:
    def __init__(self, container: _FakeContainer) -> None:
        self.containers = _FakeContainers(container)
        self.images = _FakeImages()


def _make_runner(container: _FakeContainer):
    """Import PodmanCodeRunner with a mocked Podman client."""
    module_name = "podman_code_runner"
    sys.modules.pop(module_name, None)
    client = _FakeClient(container)
    with patch("podman.PodmanClient", return_value=client):
        with patch.dict("os.environ", {"PODMAN_HOST": "http://127.0.0.1:8080"}):
            module = importlib.import_module(module_name)
            runner = module.PodmanCodeRunner()
    runner.client = client
    return runner, client


# ---------------------------------------------------------------------------
# Run behaviour
# ---------------------------------------------------------------------------

class TestPodmanCodeRunner:
    def test_successful_run_returns_stdout(self) -> None:
        runner, client = _make_runner(_FakeContainer(wait_result={"StatusCode": 0}, logs_result=b"3628800\n"))
        result = runner.run("import math\nprint(math.factorial(10))")
        assert result["success"] is True
        assert result["stdout"] == "3628800\n"
        assert result["stderr"] == ""
        assert result["exit_code"] == 0
        assert result["timed_out"] is False

    def test_failed_run_returns_stderr_and_nonzero_exit(self) -> None:
        runner, _ = _make_runner(_FakeContainer(wait_result={"StatusCode": 1}, logs_result=b"Traceback...\nValueError\n"))
        result = runner.run("raise ValueError('fail')")
        assert result["success"] is False
        assert result["exit_code"] == 1
        assert "ValueError" in result["stderr"]
        assert result["stdout"] == ""

    def test_timeout_returns_timed_out_flag(self) -> None:
        container = _FakeContainer(raise_on_wait=True)
        runner, _ = _make_runner(container)
        result = runner.run("while True: pass", timeout=10)
        assert result["success"] is False
        assert result["timed_out"] is True
        assert result["exit_code"] == -1
        assert container.killed

    def test_container_removed_after_success(self) -> None:
        container = _FakeContainer(wait_result={"StatusCode": 0}, logs_result=b"ok\n")
        runner, _ = _make_runner(container)
        runner.run("print('ok')")
        assert container.removed

    def test_container_removed_after_failure(self) -> None:
        container = _FakeContainer(wait_result={"StatusCode": 1}, logs_result=b"error\n")
        runner, _ = _make_runner(container)
        runner.run("raise Exception('fail')")
        assert container.removed

    def test_container_removed_after_timeout(self) -> None:
        container = _FakeContainer(raise_on_wait=True)
        runner, _ = _make_runner(container)
        runner.run("while True: pass", timeout=10)
        assert container.removed

    def test_uses_network_none(self) -> None:
        runner, client = _make_runner(_FakeContainer())
        runner.run("print('hi')")
        assert client.containers.run_kwargs["network_mode"] == "none"

    def test_uses_read_only_filesystem(self) -> None:
        runner, client = _make_runner(_FakeContainer())
        runner.run("print('hi')")
        assert client.containers.run_kwargs["read_only"] is True

    def test_no_volumes_mounted(self) -> None:
        runner, client = _make_runner(_FakeContainer())
        runner.run("print('hi')")
        assert "volumes" not in client.containers.run_kwargs

    def test_code_passed_via_base64_not_volume(self) -> None:
        runner, client = _make_runner(_FakeContainer())
        runner.run("print('secret')")
        command = client.containers.run_kwargs["command"]
        encoded = base64.b64encode(b"print('secret')").decode("ascii")
        assert command[:2] == ["python", "-c"]
        assert encoded in command[2]

    def test_accepts_integer_exit_status_from_podman_wait(self) -> None:
        runner, _ = _make_runner(_FakeContainer(wait_result=0, logs_result=b"result\n"))
        result = runner.run("print('result')")
        assert result["success"] is True
        assert result["stdout"] == "result\n"

    def test_accepts_chunked_logs(self) -> None:
        container = _FakeContainer()
        container.logs_result = None  # will be patched below

        runner, client = _make_runner(_FakeContainer(wait_result={"StatusCode": 0}))
        # Simulate iterator logs
        client.containers._container.logs = lambda stdout, stderr: iter((b"362", b"8800\n"))
        result = runner.run("print(3628800)")
        assert result["stdout"] == "3628800\n"

    def test_invalid_timeout_too_low_raises(self) -> None:
        runner, _ = _make_runner(_FakeContainer())
        with pytest.raises(ValueError, match="timeout"):
            runner.run("print('x')", timeout=5)

    def test_invalid_timeout_too_high_raises(self) -> None:
        runner, _ = _make_runner(_FakeContainer())
        with pytest.raises(ValueError, match="timeout"):
            runner.run("print('x')", timeout=61)

    def test_non_root_user_configured(self) -> None:
        runner, client = _make_runner(_FakeContainer())
        runner.run("print('hi')")
        assert client.containers.run_kwargs["user"] == "1000:1000"

    def test_memory_limit_set(self) -> None:
        runner, client = _make_runner(_FakeContainer())
        runner.run("print('hi')")
        assert client.containers.run_kwargs["mem_limit"] == "128m"
