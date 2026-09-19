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

    async def test_tui_layout_alignment_and_no_overflow(self):
        mock_agent = MagicMock()
        mock_agent.client.model_name = "test-model"
        mock_agent.config.enable_thinking = True
        mock_agent.get_context_status.return_value = "100k left"

        app = GemmaTUI(mock_agent)
        async with app.run_test(size=(80, 24)) as pilot:
            chat = app.query_one("#chat-container")
            inp_container = app.query_one("#input-container")
            inp = app.query_one("#user-input")
            status = app.query_one("#status-bar")
            footer = app.query_one("Footer")

            # Must align on left and right with chat container
            self.assertEqual(inp_container.region.x, chat.region.x)
            self.assertEqual(inp_container.region.width, chat.region.width)
            self.assertEqual(inp_container.region.x + inp_container.region.width, chat.region.x + chat.region.width)
            self.assertLessEqual(inp_container.region.x + inp_container.region.width, 80)

            # Input container must be strictly above status bar, not overlapping it
            self.assertLessEqual(inp_container.region.y + inp_container.region.height, status.region.y)
            # Status bar must be strictly above footer, not overlapping it
            self.assertLessEqual(status.region.y + status.region.height, footer.region.y)

    async def test_tui_layout_narrow_terminal(self):
        mock_agent = MagicMock()
        mock_agent.client.model_name = "test-model"
        mock_agent.config.enable_thinking = True
        mock_agent.get_context_status.return_value = "100k left"

        app = GemmaTUI(mock_agent)
        async with app.run_test(size=(50, 16)) as pilot:
            chat = app.query_one("#chat-container")
            inp_container = app.query_one("#input-container")
            status = app.query_one("#status-bar")

            # Must not overflow screen width (50)
            self.assertLessEqual(inp_container.region.x + inp_container.region.width, 50)
            self.assertEqual(inp_container.region.x, chat.region.x)
            self.assertEqual(inp_container.region.width, chat.region.width)
            # Must not overlap status bar
            self.assertLessEqual(inp_container.region.y + inp_container.region.height, status.region.y)


if __name__ == "__main__":
    unittest.main()

