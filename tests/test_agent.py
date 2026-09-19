import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from gemma_harness.agent import Agent, AgentConfig
from gemma_harness.tools import BASH_TOOL_DECLARATION


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # 1. Happy-path test
    def test_agent_run_turn_with_tool_call_and_compaction(self):
        mock_client = MagicMock()
        mock_client.model_name = "wyattearp/Gemma-4-26B-A4B-it-NVFP4"
        mock_client.render_prompt.return_value = "<|turn>user\nhi<turn|>"
        # Token count low initially
        mock_client.count_tokens.return_value = 100

        # Step 1: Model requests tool call
        tool_call_parsed = {
            "role": "assistant",
            "thinking": "Listing directory",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "echo 'harness test'"}
                    }
                }
            ],
            "content": ""
        }
        # Step 2: Model returns final answer
        final_answer_parsed = {
            "role": "assistant",
            "thinking": "Done",
            "tool_calls": [],
            "content": "Directory listed successfully."
        }

        mock_client.generate_completion.side_effect = [
            ("raw_tool_text", {"choices": [{"text": "raw_tool_text"}]}),
            ("raw_final_text", {"choices": [{"text": "raw_final_text"}]}),
        ]
        mock_client.parse_output.side_effect = [
            tool_call_parsed,
            final_answer_parsed
        ]

        config = AgentConfig(
            max_turns=10,
            enable_thinking=True,
            context_window=1000,
            compaction_threshold_ratio=0.85,  # 850 tokens
            transcripts_dir=self.test_dir
        )
        agent = Agent(client=mock_client, config=config)

        # Run user turn
        result = agent.run_turn("test task")
        self.assertEqual(result, "Directory listed successfully.")
        self.assertEqual(len(agent.history), 3)  # system + user + assistant

        # Test sliding window compaction
        # Add multiple dummy turns and set token count high
        for i in range(10):
            agent.history.append({"role": "user", "content": f"msg {i}"})
            agent.history.append({"role": "assistant", "content": f"resp {i}"})

        # When token count exceeds threshold
        mock_client.count_tokens.return_value = 900  # > 850 threshold
        compacted = agent.maybe_compact()
        self.assertTrue(compacted)
        # History should be reduced
        self.assertLess(len(agent.history), 22)

        # Status before compaction
        status = agent.get_context_status()
        self.assertIn("tokens", status)

    # 2. Sad-path test
    def test_agent_max_turns_and_tool_error(self):
        mock_client = MagicMock()
        mock_client.model_name = "test-model"
        mock_client.render_prompt.return_value = "<|turn>user\nloop<turn|>"
        mock_client.count_tokens.return_value = 50

        # Simulate loop: model keeps calling bash forever
        infinite_tool_parsed = {
            "role": "assistant",
            "thinking": "looping",
            "tool_calls": [
                {"function": {"name": "bash", "arguments": {"command": "echo loop"}}}
            ],
            "content": ""
        }
        mock_client.generate_completion.return_value = ("raw", {})
        mock_client.parse_output.return_value = infinite_tool_parsed

        config = AgentConfig(
            max_turns=3,  # small max turns to test limit
            enable_thinking=True,
            transcripts_dir=self.test_dir
        )
        agent = Agent(client=mock_client, config=config)

        # Should terminate with error/stop message after max turns
        with self.assertRaises(RuntimeError):
            agent.run_turn("infinite task")


if __name__ == "__main__":
    unittest.main()
