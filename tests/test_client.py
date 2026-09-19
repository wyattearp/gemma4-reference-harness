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


if __name__ == "__main__":
    unittest.main()
