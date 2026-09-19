import subprocess

BASH_TOOL_DECLARATION = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command and return its exit code, stdout, and stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command line string to execute in bash"
                }
            },
            "required": ["command"]
        }
    }
}


def execute_bash(command: str, timeout: float = 60.0) -> dict:
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        timeout_msg = f"Command timed out after {timeout} seconds"
        stderr = f"{stderr}\n{timeout_msg}".strip()
        return {
            "exit_code": 124,
            "stdout": stdout,
            "stderr": stderr
        }
    except Exception as exc:
        return {
            "exit_code": 1,
            "stdout": "",
            "stderr": str(exc)
        }
