import os
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static

from gemma_harness.agent import Agent, AgentConfig
from gemma_harness.client import GemmaClient


class GemmaTUI(App):
    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }
    #chat-container {
        height: 1fr;
        border: solid $primary;
        margin: 0 1;
    }
    #input-container {
        height: 3;
        margin: 0 1;
    }
    #user-input {
        width: 100%;
        border: solid $primary;
        padding: 0 1;
    }
    #user-input:focus {
        border: solid $accent;
    }
    #status-bar {
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_screen", "Clear Screen"),
    ]

    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, icon=">")
        with Container(id="chat-container"):
            yield RichLog(id="chat-log", wrap=True, highlight=True, markup=True)
        with Container(id="input-container"):
            yield Input(
                placeholder="Type your message or /clear, /compact, /thinking, /help, /quit...",
                id="user-input"
            )
        yield Static(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Gemma 4 Reference Harness"
        self.sub_title = f"Model: {self.agent.client.model_name or 'Auto-Detect'}"

        # Hook agent callbacks to update UI
        self.agent.on_thought = self._on_thought
        self.agent.on_tool_call = self._on_tool_call
        self.agent.on_tool_result = self._on_tool_result
        self.agent.on_content = self._on_content
        self.agent.on_status = self._on_status

        self.update_status()
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write("[bold green]Welcome to Gemma 4 Reference Harness![/bold green]")
        chat_log.write("[dim]Supports multi-turn chat, thinking mode, and bash tool execution.[/dim]")
        chat_log.write("[dim]Type [bold]/help[/bold] to see available commands.[/dim]\n")
        self.query_one("#user-input", Input).focus()

    def update_status(self, custom_status: str = None) -> None:
        status_bar = self.query_one("#status-bar", Static)
        try:
            status_text = custom_status or self.agent.get_context_status()
        except Exception:
            status_text = custom_status or "Context status unavailable"
        thinking_text = "ON" if self.agent.config.enable_thinking else "OFF"
        status_bar.update(f"[b]{status_text}[/b]  |  Thinking: [b]{thinking_text}[/b]")

    def _on_thought(self, thought: str) -> None:
        def _update():
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.write(f"[bold magenta]💭 Thinking:[/bold magenta]\n[dim]{thought}[/dim]\n")
        self.call_from_thread(_update)

    def _on_tool_call(self, tool_name: str, args: dict) -> None:
        def _update():
            chat_log = self.query_one("#chat-log", RichLog)
            cmd = args.get("command", str(args))
            chat_log.write(f"[bold yellow]⚙️ Tool Call ({tool_name}):[/bold yellow] [bold]{cmd}[/bold]")
        self.call_from_thread(_update)

    def _on_tool_result(self, tool_name: str, result: dict) -> None:
        def _update():
            chat_log = self.query_one("#chat-log", RichLog)
            stdout = result.get("stdout", "").strip()
            stderr = result.get("stderr", "").strip()
            code = result.get("exit_code", 0)
            if stdout:
                chat_log.write(f"[dim]{stdout}[/dim]")
            if stderr:
                chat_log.write(f"[red]{stderr}[/red]")
            chat_log.write(f"[dim](exit code: {code})[/dim]\n")
        self.call_from_thread(_update)

    def _on_content(self, content: str) -> None:
        def _update():
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.write(f"[bold green]Gemma 4:[/bold green]\n{content}\n")
        self.call_from_thread(_update)

    def _on_status(self, status: str) -> None:
        def _update():
            self.update_status(status)
        self.call_from_thread(_update)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""

        # Handle slash commands
        if text.startswith("/"):
            self.handle_slash_command(text)
            return

        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(f"[bold cyan]User:[/bold cyan] {text}")
        input_widget.disabled = True
        self.process_turn(text)

    def handle_slash_command(self, cmd_text: str) -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        parts = cmd_text.split()
        cmd = parts[0].lower()

        if cmd in ("/clear", "/reset"):
            self.agent.clear()
            chat_log.clear()
            chat_log.write("[yellow]Session cleared and new transcript started.[/yellow]\n")
            self.update_status()
        elif cmd in ("/compact", "/compacting"):
            compacted = self.agent.maybe_compact()
            if compacted:
                chat_log.write("[yellow]Context compacted via sliding window.[/yellow]\n")
            else:
                chat_log.write("[dim]Context not yet at compaction threshold, but status updated.[/dim]\n")
            self.update_status()
        elif cmd in ("/thinking", "/think"):
            if len(parts) > 1:
                arg = parts[1].lower()
                self.agent.config.enable_thinking = arg in ("on", "true", "1", "yes")
            else:
                self.agent.config.enable_thinking = not self.agent.config.enable_thinking
            state = "ON" if self.agent.config.enable_thinking else "OFF"
            chat_log.write(f"[yellow]Thinking mode set to {state}[/yellow]\n")
            self.update_status()
        elif cmd in ("/quit", "/exit"):
            self.exit()
        elif cmd == "/help":
            chat_log.write("[bold]Available Commands:[/bold]")
            chat_log.write("  [bold]/clear[/bold]     - Clear chat history and start fresh session")
            chat_log.write("  [bold]/compact[/bold]   - Trigger context compaction check")
            chat_log.write("  [bold]/thinking[/bold]  - Toggle thinking mode on/off (or /thinking on|off)")
            chat_log.write("  [bold]/help[/bold]      - Show this help message")
            chat_log.write("  [bold]/quit[/bold]      - Exit the application\n")
        else:
            chat_log.write(f"[red]Unknown command: {cmd}. Type /help for help.[/red]\n")

    @work(thread=True)
    def process_turn(self, user_input: str) -> None:
        try:
            self.agent.run_turn(user_input)
        except Exception as exc:
            def _show_err():
                chat_log = self.query_one("#chat-log", RichLog)
                chat_log.write(f"[bold red]Error:[/bold red] {exc}\n")
            self.call_from_thread(_show_err)
        finally:
            def _finish():
                input_widget = self.query_one("#user-input", Input)
                input_widget.disabled = False
                input_widget.focus()
                self.update_status()
            self.call_from_thread(_finish)

    def action_clear_screen(self) -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.clear()
