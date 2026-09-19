import unittest
from gemma_harness.tools import execute_bash, BASH_TOOL_DECLARATION


class TestTools(unittest.TestCase):
    # 1. Happy-path test
    def test_execute_bash_happy_path(self):
        result = execute_bash("echo 'hello world'")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"].strip(), "hello world")
        self.assertEqual(result["stderr"], "")

        # Verify BASH_TOOL_DECLARATION schema structure
        self.assertEqual(BASH_TOOL_DECLARATION["type"], "function")
        self.assertEqual(BASH_TOOL_DECLARATION["function"]["name"], "bash")
        self.assertIn("command", BASH_TOOL_DECLARATION["function"]["parameters"]["properties"])

    # 2. Sad-path test
    def test_execute_bash_error_and_timeout(self):
        # Non-zero exit code and stderr
        result = execute_bash("echo 'err' >&2 && exit 7")
        self.assertEqual(result["exit_code"], 7)
        self.assertIn("err", result["stderr"])

        # Command not found
        res_not_found = execute_bash("nonexistent_command_xyz_12345")
        self.assertNotEqual(res_not_found["exit_code"], 0)

        # Timeout handling
        res_timeout = execute_bash("sleep 2", timeout=0.1)
        self.assertNotEqual(res_timeout["exit_code"], 0)
        self.assertIn("timed out", res_timeout["stderr"].lower())


if __name__ == "__main__":
    unittest.main()
