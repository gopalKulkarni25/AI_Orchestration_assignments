"""
Podman code runner: executes Python source in a bounded, ephemeral container.

Part 2 of Assignment 2: sandboxed execution with security controls, structured
results, timeout, and cleanup.

Set PODMAN_HOST (or CONTAINER_HOST) when the Podman API is not at the default
Unix socket — required on Windows and in remote setups.
"""

from __future__ import annotations

import argparse
import base64
import os
import socket
import tarfile
from io import BytesIO
from typing import TypedDict
from urllib.parse import urlparse


class ContainerResult(TypedDict):
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


class PodmanCodeRunner:
    """Run one Python script in a temporary, tightly constrained container."""

    def __init__(self, image: str = "python:3.11-slim") -> None:
        try:
            import podman
        except ImportError as error:
            raise RuntimeError("Install dependencies from requirements.txt first.") from error

        endpoint = os.environ.get("PODMAN_HOST") or os.environ.get("CONTAINER_HOST")
        if os.name == "nt" and not endpoint:
            raise RuntimeError(
                "Podman on Windows requires PODMAN_HOST or CONTAINER_HOST. "
                "Start Podman, run 'podman system connection list', and set the variable "
                "to the default connection URI."
            )
        if (
            os.name == "nt"
            and endpoint
            and urlparse(endpoint).scheme in {"ssh", "http+ssh"}
            and not hasattr(socket, "AF_UNIX")
        ):
            raise RuntimeError(
                "The Podman Python SDK cannot use an ssh:// endpoint on this Windows Python "
                "build because it requires Unix-domain sockets. Forward the Podman machine "
                "socket to localhost with OpenSSH, then set PODMAN_HOST=http://127.0.0.1:8080."
            )

        self.client = podman.PodmanClient(base_url=endpoint) if endpoint else podman.PodmanClient()
        self.image = image

    def build_image(self, code: str, tag: str) -> None:
        """Build a reusable non-root image with app.py baked in."""
        containerfile = (
            "FROM python:3.11-slim\n"
            "WORKDIR /app\n"
            "COPY app.py /app/app.py\n"
            "USER 1000:1000\n"
            'CMD ["python", "/app/app.py"]\n'
        )
        context = BytesIO()
        with tarfile.open(fileobj=context, mode="w") as archive:
            for name, body in {"Containerfile": containerfile, "app.py": code}.items():
                encoded = body.encode("utf-8")
                entry = tarfile.TarInfo(name)
                entry.size = len(encoded)
                archive.addfile(entry, BytesIO(encoded))
        context.seek(0)
        self.client.images.build(
            fileobj=context,
            custom_context=True,
            dockerfile="Containerfile",
            tag=tag,
            pull=False,
        )

    def run(self, code: str, timeout: int = 20) -> ContainerResult:
        """Execute code in an ephemeral container with all security controls applied."""
        if not 10 <= timeout <= 60:
            raise ValueError(f"timeout must be between 10 and 60 seconds, got {timeout}")

        container = None
        # Pass source through base64 to avoid host path issues on Windows (where the
        # host path is not a valid path inside the Podman Linux VM).
        encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
        try:
            container = self.client.containers.run(
                image=self.image,
                command=[
                    "python",
                    "-c",
                    "import base64; exec(compile(base64.b64decode("
                    + repr(encoded_code)
                    + "), 'script.py', 'exec'))",
                ],
                detach=True,
                network_mode="none",    # no outbound or inbound network access
                read_only=True,         # immutable root filesystem
                mem_limit="128m",       # 128 MB hard memory cap
                cpu_period=100_000,     # 50% CPU quota (50k / 100k)
                cpu_quota=50_000,
                user="1000:1000",       # non-root UID/GID
            )
            try:
                outcome = container.wait(timeout=timeout)
            except Exception:
                container.kill()
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Execution timed out.",
                    "exit_code": -1,
                    "timed_out": True,
                }

            # Podman API versions return either a Docker-compatible dict or a plain int.
            exit_code = int(outcome["StatusCode"] if isinstance(outcome, dict) else outcome)
            log_output = container.logs(stdout=True, stderr=True)
            log_bytes = log_output if isinstance(log_output, bytes) else b"".join(log_output)
            logs = log_bytes.decode("utf-8", errors="replace")
            return {
                "success": exit_code == 0,
                "stdout": logs if exit_code == 0 else "",
                "stderr": "" if exit_code == 0 else logs,
                "exit_code": exit_code,
                "timed_out": False,
            }
        except Exception as error:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Sandbox error: {type(error).__name__}: {error}",
                "exit_code": -1,
                "timed_out": False,
            }
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Podman code runner demonstrations")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: print math.factorial(10).")
    parser.add_argument("--fail", action="store_true", help="Failure test: raise an exception, return stderr.")
    parser.add_argument("--timeout", action="store_true", help="Timeout test: infinite loop, safely stopped.")
    parser.add_argument("--build", action="store_true", help="Build a reusable image after a successful run.")
    parser.add_argument("--image", default="student-sandbox-app:latest", help="Image tag for --build (default: student-sandbox-app:latest).")
    args = parser.parse_args()

    if not any([args.smoke, args.fail, args.timeout, args.build]):
        parser.error("Choose at least one of --smoke, --fail, --timeout, --build.")

    try:
        runner = PodmanCodeRunner()
    except RuntimeError as error:
        parser.error(str(error))

    if args.smoke:
        print("--- Smoke test ---")
        result = runner.run("import math\nprint(math.factorial(10))")
        print(result)

    if args.fail:
        print("--- Failure test ---")
        result = runner.run("raise ValueError('intentional failure')")
        print(result)

    if args.timeout:
        print("--- Timeout test ---")
        result = runner.run("while True: pass", timeout=10)
        print(result)

    if args.build:
        print("--- Build test ---")
        code = "import math\nprint(math.factorial(10))"
        result = runner.run(code)
        print(f"Run result: {result}")
        if result["success"]:
            runner.build_image(code, args.image)
            print(f"Image built: {args.image}")
            print(f"Run with: podman run --rm {args.image}")
        else:
            print("Build skipped: run was not successful.")


if __name__ == "__main__":
    main()
