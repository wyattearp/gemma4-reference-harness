import unittest
from unittest.mock import MagicMock
from gemma_harness.tui import GemmaTUI
from textual.widgets import Header, Static


class TestTUI(unittest.IsolatedAsyncioTestCase):
    # 1. Happy-path test (Red: currently Header uses \u2b58)
    async def test_header_icon_uses_safe_ascii(self):
        mock_agent = MagicMock()
        mock_agent.client.model_name = "test-model"
        mock_agent.config.enable_thinking = True
        mock_agent.get_context_status.return_value = "100,000 tokens left"

        app = GemmaTUI(mock_agent)
        async with app.run_test():
            header = app.query_one(Header)
            # Must NOT use the broken Unicode U+2B58 ('⭘')
            self.assertNotEqual(header.icon, "\u2b58")
            self.assertEqual(header.icon, ">")

    # 2. Sad-path test (handles empty/error status safely)
    async def test_tui_status_update_edge_cases(self):
        mock_agent = MagicMock()
        mock_agent.client.model_name = ""
        mock_agent.config.enable_thinking = False
        mock_agent.get_context_status.side_effect = RuntimeError("Context estimation failure")

        app = GemmaTUI(mock_agent)
        async with app.run_test():
            # When context status fails, update_status should fall back safely without crashing
            app.update_status(custom_status="Error fallback")
            status_bar = app.query_one("#status-bar", Static)
            self.assertIn("Error fallback", str(status_bar.content))


if __name__ == "__main__":
    unittest.main()
