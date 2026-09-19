import json
import unittest
from unittest.mock import patch, MagicMock
import urllib.error

from gemma_harness.client import GemmaClient
from gemma_harness.tools import BASH_TOOL_DECLARATION


class TestGemmaClient(unittest.TestCase):
    # 1. Happy-path test
    @patch("urllib.request.urlopen")
    def test_client_happy_path(self, mock_urlopen):
        # Mock /v1/models response
        mock_models_resp = MagicMock()
        mock_models_resp.read.return_value = json.dumps({
            "data": [
                {"id": "gpt-4o"},
                {"id": "wyattearp/Gemma-4-26B-A4B-it-NVFP4"}
            ]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_models_resp

        client = GemmaClient(base_url="http://mock-server/v1", api_key="test-key")
        model_id = client.discover_model()
        self.assertEqual(model_id, "wyattearp/Gemma-4-26B-A4B-it-NVFP4")

        # Test prompt formatting
        messages = [
            {"role": "user", "content": "list directory"}
        ]
        prompt = client.render_prompt(messages, tools=[BASH_TOOL_DECLARATION], enable_thinking=True)
        self.assertIn("<|turn>user", prompt)
        self.assertIn("declaration:bash", prompt)
        self.assertIn("<|think|>", prompt)

        # Test response parsing
        raw_text = '<|channel>thought\nChecking files...<channel|><|tool_call>call:bash{command:<|"|>ls -la<|"|>}<tool_call|>'
        parsed = client.parse_output(raw_text, prefix=prompt, tools=[BASH_TOOL_DECLARATION])
        self.assertEqual(parsed["thinking"], "Checking files...")
        self.assertEqual(len(parsed["tool_calls"]), 1)
        self.assertEqual(parsed["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(parsed["tool_calls"][0]["function"]["arguments"], {"command": "ls -la"})

        # Test response parsing with secondary leaked thought channel (HF parser limitation test)
        leaked_raw = (
            "<|channel>thought\n<channel|>Initial reasoning thought here.\n<channel|>"
            "This is the actual final response."
        )
        parsed_leak = client.parse_output(leaked_raw, prefix=prompt, tools=[BASH_TOOL_DECLARATION])
        self.assertEqual(parsed_leak["thinking"], "Initial reasoning thought here.")
        self.assertEqual(parsed_leak["content"], "This is the actual final response.")

        # Mock /v1/completions response
        mock_compl_resp = MagicMock()
        mock_compl_resp.read.return_value = json.dumps({
            "choices": [{"text": raw_text, "finish_reason": "stop"}]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_compl_resp

        result_text, raw_dict = client.generate_completion(prompt, max_tokens=128)
        self.assertEqual(result_text, raw_text)

    # 2. Sad-path test
    @patch("urllib.request.urlopen")
    def test_client_errors_and_edge_cases(self, mock_urlopen):
        # 1. Error when /models has no Gemma model
        mock_models_resp = MagicMock()
        mock_models_resp.read.return_value = json.dumps({
            "data": [{"id": "some-other-model"}]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_models_resp

        client = GemmaClient(base_url="http://mock-server/v1", api_key="test-key")
        # Should raise ValueError or fallback
        with self.assertRaises(ValueError):
            client.discover_model(require_gemma=True)

        # 2. HTTP error during completion
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://mock-server/v1/completions",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=MagicMock(read=lambda: b"Server exploded")
        )
        with self.assertRaises(RuntimeError):
            client.generate_completion("some prompt")

    # Happy-path: stop tokens include tool delimiters
    @patch("urllib.request.urlopen")
    def test_generate_completion_stop_tokens(self, mock_urlopen):
        mock_compl_resp = MagicMock()
        mock_compl_resp.read.return_value = json.dumps({
            "choices": [{"text": "ok", "finish_reason": "stop"}]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_compl_resp

        client = GemmaClient(base_url="http://mock-server/v1", api_key="test-key", model_name="gemma-4")
        client.generate_completion("prompt")

        args, kwargs = mock_urlopen.call_args
        req = args[0]
        payload = json.loads(req.data.decode("utf-8"))
        self.assertIn("<tool_call|>", payload["stop"])
        self.assertIn("<tool_response|>", payload["stop"])

    # Sad-path / edge-case: malformed closing tag or stripped closing tag
    def test_parse_output_mismatched_and_stripped_tool_call_closing_tags(self):
        client = GemmaClient(base_url="http://mock-server/v1", api_key="test-key", model_name="wyattearp/Gemma-4-26B-A4B-it-NVFP4")
        malformed = (
            '<|channel>thought\nThinking...\n<channel|>'
            '<|tool_call>call:bash{command:<|"|>cat << \'EOF\' > snake.py\ncode\nEOF\n<|"|>}<tool_response|>'
            '<|channel>thought\n<channel|><|tool_call>call:bash{command:<|"|>ls -la<|"|>}<tool_response|>'
        )
        parsed = client.parse_output(malformed, prefix="", tools=[BASH_TOOL_DECLARATION])
        self.assertIn("tool_calls", parsed)
        self.assertGreaterEqual(len(parsed["tool_calls"]), 1)
        self.assertEqual(parsed["tool_calls"][0]["function"]["name"], "bash")
        self.assertIn("cat << 'EOF' > snake.py", parsed["tool_calls"][0]["function"]["arguments"]["command"])

        # Also test stripped closing tag (when stop sequence chops off <tool_call|>)
        stripped = '<|tool_call>call:bash{command:<|"|>python3 out.py<|"|>}'
        parsed_stripped = client.parse_output(stripped, prefix="", tools=[BASH_TOOL_DECLARATION])
        self.assertIn("tool_calls", parsed_stripped)
        self.assertEqual(len(parsed_stripped["tool_calls"]), 1)
        self.assertEqual(parsed_stripped["tool_calls"][0]["function"]["arguments"]["command"], "python3 out.py")

    def test_get_tokenizer_patches_response_template(self):
        client = GemmaClient(base_url="http://mock-server/v1", api_key="test-key", model_name="wyattearp/Gemma-4-26B-A4B-it-NVFP4")
        tok = client.get_tokenizer()
        self.assertIsNotNone(tok.response_template)
        close_delims = tok.response_template["fields"]["tool_calls"]["close"]
        self.assertIn("<tool_call|>", close_delims)
        self.assertIn("<tool_response|>", close_delims)

    def test_parse_response_natively_parses_tool_response_closing_tag(self):
        client = GemmaClient(base_url="http://mock-server/v1", api_key="test-key", model_name="wyattearp/Gemma-4-26B-A4B-it-NVFP4")
        tok = client.get_tokenizer()
        s = '<|tool_call>call:bash{command:<|"|>echo test<|"|>}<tool_response|>'
        tools = [BASH_TOOL_DECLARATION]
        res = tok.parse_response(s, prefix="", tools=tools)
        self.assertEqual(res["tool_calls"][0]["function"]["arguments"]["command"], "echo test")


    # 3. Warning suppression tests
    def test_library_warnings_silenced(self):
        import subprocess
        import sys
        res = subprocess.run(
            [sys.executable, "-c", "from gemma_harness.client import GemmaClient; client = GemmaClient(); client.get_tokenizer()"],
            capture_output=True,
            text=True
        )
        self.assertNotIn("PyTorch was not found", res.stderr)
        self.assertNotIn("unauthenticated requests", res.stderr)

    def test_library_warnings_with_explicit_env(self):
        import subprocess
        import sys
        import os
        env = os.environ.copy()
        env["TRANSFORMERS_VERBOSITY"] = "info"
        res = subprocess.run(
            [sys.executable, "-c", "import os; from gemma_harness.client import GemmaClient; print(os.environ.get('TRANSFORMERS_VERBOSITY'))"],
            capture_output=True,
            text=True,
            env=env
        )
        self.assertEqual(res.stdout.strip(), "info")


if __name__ == "__main__":
    unittest.main()
