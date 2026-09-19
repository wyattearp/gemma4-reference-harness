import os
import shutil
import tempfile
import unittest
from gemma_harness.sandbox import DockerSandbox


class TestDockerSandbox(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_sandbox_ws_")
        self.sandbox = DockerSandbox(
            image="gemma4-sandbox:latest",
            workspace_dir=self.test_dir,
            memory="2g"
        )

    def tearDown(self):
        self.sandbox.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # 1. Happy-path tests
    def test_sandbox_execute_success(self):
        result = self.sandbox.execute("echo 'hello sandbox'")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"].strip(), "hello sandbox")
        self.assertEqual(result["stderr"], "")

    def test_sandbox_workspace_file_persisted(self):
        result = self.sandbox.execute("echo 'file content' > test_persisted.txt")
        self.assertEqual(result["exit_code"], 0)

        host_file = os.path.join(self.test_dir, "test_persisted.txt")
        self.assertTrue(os.path.exists(host_file))
        with open(host_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content.strip(), "file content")

        # Verify host ownership (non-root)
        stat_info = os.stat(host_file)
        self.assertEqual(stat_info.st_uid, os.getuid())

    def test_sandbox_rfc1918_routes_configured(self):
        # Verify routing table has unreachable routes for RFC 1918 subnets
        result = self.sandbox.execute("ip route")
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("10.0.0.0/8", result["stdout"])
        self.assertIn("172.16.0.0/12", result["stdout"])
        self.assertIn("192.168.0.0/16", result["stdout"])

    # 2. Sad-path tests
    def test_sandbox_command_timeout(self):
        result = self.sandbox.execute("sleep 2", timeout=0.5)
        self.assertEqual(result["exit_code"], 124)
        self.assertIn("timed out after 0.5 seconds", result["stderr"])

    def test_sandbox_command_error(self):
        result = self.sandbox.execute("ls /definitely_does_not_exist_xyz123")
        self.assertNotEqual(result["exit_code"], 0)
        self.assertIn("No such file or directory", result["stderr"])

    def test_sandbox_idempotent_stop(self):
        # Stopping unstarted or stopped sandbox should never raise
        self.sandbox.stop()
        self.sandbox.stop()

    def test_sandbox_invalid_image_fails_loudly(self):
        bad_sandbox = DockerSandbox(
            image="nonexistent_image_123456789:bad_tag",
            workspace_dir=self.test_dir,
            auto_build=False
        )
        with self.assertRaises(RuntimeError):
            bad_sandbox.start()


if __name__ == "__main__":
    unittest.main()
