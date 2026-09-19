import unittest
from unittest.mock import patch, MagicMock
from gemma_harness.cli import build_parser, run_cli


class TestCLI(unittest.TestCase):
    # 1. Happy-path test
    def test_cli_parser_happy_path(self):
        parser = build_parser()
        args = parser.parse_args(["-p", "test prompt", "--max-turns", "50", "--no-thinking"])
        self.assertEqual(args.prompt, "test prompt")
        self.assertEqual(args.max_turns, 50)
        self.assertFalse(args.thinking)

    @patch("gemma_harness.cli.Agent")
    @patch("gemma_harness.cli.GemmaClient")
    def test_run_cli_one_shot(self, mock_client_cls, mock_agent_cls):
        mock_client = MagicMock()
        mock_client.discover_model.return_value = "gemma-4"
        mock_client_cls.return_value = mock_client

        mock_agent = MagicMock()
        mock_agent.run_turn.return_value = "Task complete!"
        mock_agent_cls.return_value = mock_agent

        exit_code = run_cli(["-p", "do something"])
        self.assertEqual(exit_code, 0)
        mock_agent.run_turn.assert_called_once_with("do something")

    # 2. Sad-path test
    @patch("gemma_harness.cli.Agent")
    @patch("gemma_harness.cli.GemmaClient")
    def test_run_cli_failure(self, mock_client_cls, mock_agent_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_agent = MagicMock()
        mock_agent.run_turn.side_effect = RuntimeError("Max turns exceeded")
        mock_agent_cls.return_value = mock_agent

        exit_code = run_cli(["-p", "fail task"])
        self.assertEqual(exit_code, 1)

    def test_cli_parser_sandbox_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.sandbox, "docker")
        self.assertEqual(args.sandbox_image, "gemma4-sandbox:latest")
        self.assertEqual(args.workspace_dir, "./workspace")

    def test_cli_parser_sandbox_custom(self):
        parser = build_parser()
        args = parser.parse_args(["--sandbox", "none", "--workspace-dir", "/tmp/ws", "--sandbox-image", "my-img:v1"])
        self.assertEqual(args.sandbox, "none")
        self.assertEqual(args.sandbox_image, "my-img:v1")
        self.assertEqual(args.workspace_dir, "/tmp/ws")

    def test_cli_parser_invalid_sandbox_choice(self):
        parser = build_parser()
        # argparse prints error and exits with code 2 on invalid choices
        with self.assertRaises(SystemExit):
            with patch("sys.stderr"):
                parser.parse_args(["--sandbox", "invalid-mode"])

    def test_cli_system_prompt_arguments(self):
        parser = build_parser()
        args = parser.parse_args(["--system-prompt", "custom instruction", "--system-prompt-file", "my_prompt.txt"])
        self.assertEqual(args.system_prompt, "custom instruction")
        self.assertEqual(args.system_prompt_file, "my_prompt.txt")

    @patch("gemma_harness.cli.GemmaClient")
    def test_cli_system_prompt_file_missing_exits_with_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        exit_code = run_cli(["--system-prompt-file", "/nonexistent/prompt_file_987.txt", "-p", "hello"])
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()


