# Assignment 2 — Validated MCP Tools and Bounded Podman Execution

A Python application combining a typed, validated FastMCP file workspace with an ephemeral, security-hardened Podman code runner.

---

## Architecture

```
validated_file_service.py
  ├── WorkspaceManager       — bounded filesystem operations
  ├── Pydantic models        — WriteFile, ReadFile, ListDirectory
  ├── FileToolRegistry       — single validation/dispatch boundary
  └── FastMCP server         — write_file, read_file, list_files + workspace://files resource

podman_code_runner.py
  ├── ContainerResult        — TypedDict: success, stdout, stderr, exit_code, timed_out
  └── PodmanCodeRunner       — run(), build_image() with all security controls
```

---

## Prerequisites

- Python 3.11+
- Podman Desktop or Podman CLI (`podman --version`)
- On **Windows**, set `PODMAN_HOST` to the Podman API endpoint:
  ```
  podman system connection list
  # Then forward the socket via OpenSSH and set:
  PODMAN_HOST=http://127.0.0.1:8080
  ```

---

## Setup

```bash
pip install -r requirements.txt
```

Pull the base image once before running the sandbox:

```bash
podman pull python:3.11-slim
```

---

## Running the Demonstrations

> None of the required demonstrations require an LLM or API key.

### 1. FastMCP discovery, valid write/read, and traversal rejection

```bash
python validated_file_service.py --explore
```

Expected output shows:
- Three discovered tools (`write_file`, `read_file`, `list_files`)
- The `workspace://files` resource
- Successful write and read of `demo.txt`
- A validation error when attempting `../evil.txt`

### 2. Smoke test — `math.factorial(10)`

```bash
python podman_code_runner.py --smoke
```

### 3. Failure test — exception with stderr and non-zero exit code

```bash
python podman_code_runner.py --fail
```

### 4. Timeout test — infinite loop safely stopped

```bash
python podman_code_runner.py --timeout
```

### 5. Build a reusable image, then run it

```bash
python podman_code_runner.py --build
podman run --rm student-sandbox-app:latest
```

---

## Running Tests

```bash
python -m pytest
```

With coverage:

```bash
python -m pytest --cov=validated_file_service --cov=podman_code_runner --cov-report=term-missing
```

The `test_podman_code_runner.py` tests use a mocked Podman client and do **not** require a running Podman daemon.

---

## Security Controls (Podman sandbox)

| Control | Value |
|---|---|
| Network | Disabled (`network_mode="none"`) |
| Root filesystem | Read-only |
| Memory limit | 128 MB |
| CPU quota | 50% (50 000 / 100 000) |
| Container user | UID 1000:1000 (non-root) |
| Execution timeout | 10 – 60 seconds |
| Host mounts | None (code passed via base64) |
| Container cleanup | `finally` block — always removed |

---

## Project Structure

```
ai-orchestrator-assignment-2/
├── validated_file_service.py      # Part 1: FastMCP + Pydantic registry
├── podman_code_runner.py          # Part 2: Podman sandbox runner
├── test_validated_file_service.py # pytest — validation and registry paths
├── test_podman_code_runner.py     # pytest — container behaviour (mocked)
├── requirements.txt
├── README.md
└── workspace/                     # Created on first run; managed by WorkspaceManager
```
