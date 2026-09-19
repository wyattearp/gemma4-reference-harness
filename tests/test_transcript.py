import os
import shutil
import tempfile
import unittest
from datetime import datetime
from gemma_harness.transcript import TranscriptLogger


class TestTranscript(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # 1. Happy-path test
    def test_transcript_logging_and_file_creation(self):
        logger = TranscriptLogger(base_dir=self.test_dir, model_name="test-gemma-4")
        logger.log_prompt(prompt="<|turn>user\nhi<turn|>", token_count=5)
        logger.log_raw_completion(raw_text="hello", token_count=2, stop_reason="<turn|>")
        logger.log_tool_call(tool_name="bash", arguments={"command": "ls"}, result={"exit_code": 0, "stdout": "ok"})

        file_path = logger.save()
        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(file_path.endswith(".jsonc"))

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("// Gemma 4 Reference Harness Transcript", content)
        self.assertIn("test-gemma-4", content)
        self.assertIn("bash", content)
        self.assertIn("ls", content)

    # 2. Sad-path test
    def test_transcript_non_serializable_and_empty(self):
        logger = TranscriptLogger(base_dir=self.test_dir, model_name="test-gemma-4")
        # Non-serializable object passed in arguments or result
        class UnserializableObject:
            pass

        logger.log_tool_call(
            tool_name="bash",
            arguments={"obj": UnserializableObject()},
            result={"raw": object()}
        )
        file_path = logger.save()
        self.assertTrue(os.path.exists(file_path))

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("UnserializableObject", content)


if __name__ == "__main__":
    unittest.main()
