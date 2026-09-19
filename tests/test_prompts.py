import os
import tempfile
import unittest

from gemma_harness.agent import AgentConfig
from gemma_harness.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    get_prompts_dir,
    load_prompt,
    load_system_prompt,
)


class TestPrompts(unittest.TestCase):
    # 1. Happy-path tests
    def test_load_system_prompt_default(self):
        prompt = load_system_prompt()
        self.assertIsInstance(prompt, str)
        self.assertTrue(len(prompt) > 0)
        self.assertIn("helpful assistant with access to a bash tool", prompt)

    def test_load_custom_prompt_file(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            f.write("You are a specialized math solver.")
            temp_path = f.name
        try:
            loaded = load_prompt(temp_path)
            self.assertEqual(loaded, "You are a specialized math solver.")
        finally:
            os.remove(temp_path)

    def test_agent_config_loads_system_prompt_by_default(self):
        config = AgentConfig()
        self.assertEqual(config.system_prompt, load_system_prompt())

    # 2. Sad-path tests
    def test_load_prompt_nonexistent_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            load_prompt("nonexistent_prompt_file_xyz123.txt")

    def test_load_prompt_with_fallback(self):
        fallback_val = "Emergency fallback instructions"
        res = load_prompt("nonexistent_prompt_file_xyz123.txt", fallback=fallback_val)
        self.assertEqual(res, fallback_val)


if __name__ == "__main__":
    unittest.main()
