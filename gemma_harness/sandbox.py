import atexit
import os
import subprocess
from typing import Dict, Optional


class DockerSandbox:
    def __init__(
        self,
        image: str = "gemma4-sandbox:latest",
        workspace_dir: str = "./workspace",
        memory: str = "4g",
        pids_limit: int = 256,
        runtime: str = "runsc",
        auto_build: bool = True,
    ):
        self.image = image
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.memory = memory
        self.pids_limit = pids_limit
        self.runtime = runtime
        self.auto_build = auto_build
        self.container_name = f"gemma-sandbox-{os.getpid()}"
        self._started = False
        atexit.register(self.stop)

    def ensure_image(self) -> None:
        inspect = subprocess.run(
            ["docker", "image", "inspect", self.image],
            capture_output=True,
        )
        if inspect.returncode == 0:
            return

        if not self.auto_build:
            raise RuntimeError(f"Sandbox image {self.image!r} does not exist locally and auto_build=False")

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dockerfile = os.path.join(repo_root, "Dockerfile.sandbox")
        if not os.path.exists(dockerfile):
            raise RuntimeError(f"Cannot build sandbox image {self.image!r}: {dockerfile} not found")

        build_res = subprocess.run(
            ["docker", "build", "-t", self.image, "-f", dockerfile, repo_root],
            capture_output=True,
            text=True,
        )
        if build_res.returncode != 0:
            raise RuntimeError(f"Failed to build sandbox image {self.image!r}: {build_res.stderr.strip()}")

    def start(self) -> None:
        if self._started:
            return

        self.ensure_image()
        os.makedirs(self.workspace_dir, exist_ok=True)

        # Remove any lingering container with our name
        subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)

        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "--runtime", self.runtime,
            "--cap-add", "NET_ADMIN",
            "-v", f"{self.workspace_dir}:/workspace",
            "-w", "/workspace",
            "--memory", self.memory,
            "--pids-limit", str(self.pids_limit),
            self.image,
            "sleep", "infinity",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to start Docker sandbox: {res.stderr.strip()}")

        # Block RFC 1918 private subnets so container cannot access host or local LAN
        route_cmd = (
            "ip route add unreachable 10.0.0.0/8 && "
            "ip route add unreachable 172.16.0.0/12 && "
            "ip route add unreachable 192.168.0.0/16"
        )
        route_res = subprocess.run(
            ["docker", "exec", self.container_name, "bash", "-c", route_cmd],
            capture_output=True,
            text=True,
        )
        if route_res.returncode != 0:
            self.stop()
            raise RuntimeError(f"Failed to configure sandbox routing: {route_res.stderr.strip()}")

        self._started = True

    def execute(self, command: str, timeout: float = 60.0) -> Dict[str, any]:
        if not self._started:
            self.start()

        # Run command with host user ID so created files are owned by host user
        uid = os.getuid()
        gid = os.getgid()

        try:
            proc = subprocess.run(
                [
                    "docker", "exec",
                    "-u", f"{uid}:{gid}",
                    self.container_name,
                    "bash", "-c", command,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            timeout_msg = f"Command timed out after {timeout} seconds"
            stderr = f"{stderr}\n{timeout_msg}".strip()
            return {
                "exit_code": 124,
                "stdout": stdout,
                "stderr": stderr,
            }
        except Exception as exc:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": str(exc),
            }

    def stop(self) -> None:
        if self._started:
            subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)
            self._started = False
