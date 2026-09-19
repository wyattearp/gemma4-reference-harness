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
            ("raw_tool_text", {"choices": [{"text": "raw_tool_text", "finish_reason": "tool_calls"}]}),
            ("raw_final_text", {"choices": [{"text": "raw_final_text", "finish_reason": "stop"}]}),
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

        raw_events = [e for e in agent.transcript.events if e["type"] == "raw_completion_received"]
        self.assertEqual(len(raw_events), 2)
        self.assertEqual(raw_events[0]["stop_reason"], "tool_calls")
        self.assertEqual(raw_events[1]["stop_reason"], "stop")

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

        raw_events = [e for e in agent.transcript.events if e["type"] == "raw_completion_received"]
        self.assertTrue(len(raw_events) > 0)
        self.assertIsNone(raw_events[0]["stop_reason"])

    # Truncation guard test
    from unittest.mock import patch
    @patch("gemma_harness.agent.execute_bash")
    def test_agent_truncation_guard_aborts_cut_off_command(self, mock_bash):
        mock_client = MagicMock()
        mock_client.model_name = "test-model"
        mock_client.render_prompt.return_value = "<|turn>user\nhi<turn|>"
        mock_client.count_tokens.return_value = 100

        truncated_raw = "<|tool_call>call:bash{command:<|\"|>cat << 'EOF' > snake.py\ncode..."
        mock_client.generate_completion.side_effect = [
            (truncated_raw, {"choices": [{"text": truncated_raw, "finish_reason": "length"}]}),
            ("Recovered", {"choices": [{"text": "Recovered", "finish_reason": "stop"}]}),
        ]
        mock_client.parse_output.side_effect = [
            {"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "cat << 'EOF' > snake.py\ncode..."}}}], "content": ""},
            {"role": "assistant", "tool_calls": [], "content": "Recovered"}
        ]

        agent = Agent(client=mock_client, config=AgentConfig(max_turns=3, transcripts_dir=self.test_dir))
        result = agent.run_turn("write snake")

        # mock_bash should NOT have been called with the truncated command!
        mock_bash.assert_not_called()
        self.assertEqual(result, "Recovered")

    def test_repetition_penalty_default(self):
        self.assertEqual(AgentConfig().repetition_penalty, 1.15)

    # Sandbox integration tests
    def test_agent_executes_via_sandbox(self):
        mock_client = MagicMock()
        mock_client.model_name = "test-model"
        mock_client.render_prompt.return_value = "<|turn>user\nhi<turn|>"
        mock_client.count_tokens.return_value = 100
        mock_client.generate_completion.side_effect = [
            ("<|tool_call>call:bash{command:<|\"|>echo hi<|\"|>}", {"choices": [{"text": "call", "finish_reason": "stop"}]}),
            ("Done!", {"choices": [{"text": "Done!", "finish_reason": "stop"}]}),
        ]
        mock_client.parse_output.side_effect = [
            {"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "echo hi"}}}], "content": ""},
            {"role": "assistant", "tool_calls": [], "content": "Done!"}
        ]

        mock_sandbox = MagicMock()
        mock_sandbox.execute.return_value = {"exit_code": 0, "stdout": "hi\n", "stderr": ""}

        agent = Agent(
            client=mock_client,
            config=AgentConfig(max_turns=3, sandbox_mode="docker", transcripts_dir=self.test_dir)
        )
        agent.sandbox = mock_sandbox

        res = agent.run_turn("run echo")
        self.assertEqual(res, "Done!")
        mock_sandbox.execute.assert_called_once_with("echo hi")

    @patch("gemma_harness.agent.execute_bash")
    def test_agent_executes_fallback_when_sandbox_none(self, mock_bash):
        mock_bash.return_value = {"exit_code": 0, "stdout": "local hi\n", "stderr": ""}

        mock_client = MagicMock()
        mock_client.model_name = "test-model"
        mock_client.render_prompt.return_value = "<|turn>user\nhi<turn|>"
        mock_client.count_tokens.return_value = 100
        mock_client.generate_completion.side_effect = [
            ("<|tool_call>call:bash{command:<|\"|>echo local<|\"|>}", {"choices": [{"text": "call", "finish_reason": "stop"}]}),
            ("Done!", {"choices": [{"text": "Done!", "finish_reason": "stop"}]}),
        ]
        mock_client.parse_output.side_effect = [
            {"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "echo local"}}}], "content": ""},
            {"role": "assistant", "tool_calls": [], "content": "Done!"}
        ]

        agent = Agent(
            client=mock_client,
            config=AgentConfig(max_turns=3, sandbox_mode="none", transcripts_dir=self.test_dir)
        )

        res = agent.run_turn("run local")
        self.assertEqual(res, "Done!")
        mock_bash.assert_called_once_with("echo local")

    def test_agent_extracts_positional_and_markdown_wrapped_command(self):
        mock_client = MagicMock()
        mock_client.model_name = "test-model"
        mock_client.render_prompt.return_value = "<|turn>user\nhi<turn|>"
        mock_client.count_tokens.return_value = 100

        mock_sandbox = MagicMock()
        mock_sandbox.execute.return_value = {"exit_code": 0, "stdout": "extracted\n", "stderr": ""}

        agent = Agent(
            client=mock_client,
            config=AgentConfig(max_turns=3, sandbox_mode="docker", transcripts_dir=self.test_dir)
        )
        agent.sandbox = mock_sandbox

        # Case 1: Key "1" with conversational preamble and markdown code block
        args_with_markdown = {
            "1": "Generating the script to create snake.py:\n\n```bash\ncat << 'EOF' > snake.py\ncode\nEOF\n```"
        }
        cmd = agent.extract_command(args_with_markdown)
        self.assertEqual(cmd, "cat << 'EOF' > snake.py\ncode\nEOF")

        # Case 2: Key "0" plain command
        cmd_zero = agent.extract_command({"0": "ls -la"})
        self.assertEqual(cmd_zero, "ls -la")

    def test_agent_extracts_command_sad_path(self):
        agent = Agent(client=MagicMock(), config=AgentConfig(sandbox_mode="none", transcripts_dir=self.test_dir))
        self.assertEqual(agent.extract_command({}), "")
        self.assertEqual(agent.extract_command(None), "")
        self.assertEqual(agent.extract_command(12345), "")
        self.assertEqual(agent.extract_command({"foo": "bar", "baz": "qux"}), "")


if __name__ == "__main__":
    unittest.main()


