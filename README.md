# Gemma 4 Reference Harness

A reference implementation and agent harness for **Gemma 4** (`wyattearp/Gemma-4-26B-A4B-it-NVFP4` and related models). Designed following the **Caveman Flat** philosophy (flat procedural logic, minimal abstractions, standard library first).

---

## Key Gemma 4 Specifications & Findings

Gemma 4 introduces a new turn and tool-calling token schema:
- Turn delimiters: `<|turn>system`, `<|turn>user`, `<|turn>model`, with turn ending `<turn|>`.
- Thinking mode: `<|think|>` token in the system turn enables reasoning in `<|channel>thought\n...\n<channel|>`.
- Tool declarations: `<|tool>declaration:name{...}<tool|>` in the system turn.
- Tool requests: `<|tool_call>call:name{arg:<|"|>val<|"|>}<tool_call|>`.
- Tool responses: `<|tool_response>response:name{...}<tool_response|>`.

### vLLM Integration Solution
- Standard vLLM `/v1/chat/completions` strips `<|channel>thought` internal reasoning channels when tools are invoked.
- Standard vLLM `/v1/completions` defaults to `skip_special_tokens=True`, which strips `<|tool_call>` (token 48) and `<tool_call|>` (token 49), breaking Hugging Face's response parser regex.
- **Solution in this harness**: Dispatches raw chat-templated prompts to `/v1/completions` with `skip_special_tokens=False` and stop sequences `["<turn|>", "<|tool_response>"]`. Both thoughts and structured function calls are parsed faithfully using Hugging Face `transformers` tokenizers and response parsers.

---

## Features

1. **Multi-Turn Agent with Bash Tool**:
   - Executes shell commands, capturing exit codes, stdout, and stderr.
   - Preserves reasoning during same-turn tool handshakes while stripping thoughts on subsequent user turns per Google's specification.

2. **256K Context Window & Sliding-Window Compaction**:
   - Gemma 4 supports up to 256K tokens (262,144 tokens).
   - At ~85% capacity (~222,822 tokens), the agent automatically compacts history via a sliding window, keeping system instructions and recent context.
   - Status bar displays remaining tokens before compaction.

3. **Two Operating Modes**:
   - **Interactive Mode**: Full Textual TUI with scrollable thought & response log, input bar, context tracker, and slash commands (`/clear`, `/compact`, `/thinking`, `/help`, `/quit`).
   - **One-Shot Mode**: Non-interactive CLI flag (`-p "prompt"`, `--max-turns 100`).

4. **Transcript Logging**:
   - Automatically writes every prompt, raw completion, tool execution, and compaction event to `./transcripts/{isodate}.jsonc` (with `./trasncripts` symlink support).

---

## Quick Start

### 1. Requirements & Setup
The harness uses Python 3.10+, `transformers`, `textual`, and `jinja2`.

```bash
# Activate virtual environment
source .venv/bin/activate
```

### 2. Run Interactive Mode (Textual TUI)
```bash
python -m gemma_harness
```

#### Interactive Commands:
- `/clear` - Reset conversation history and start fresh transcript.
- `/compact` - Manually trigger sliding-window context compaction.
- `/thinking [on|off]` - Toggle thinking mode.
- `/help` - Show available commands.
- `/quit` - Exit the application.

### 3. Run One-Shot Mode
```bash
python -m gemma_harness -p "What is the kernel version and uptime of this machine? Use bash to check."
```

Optional flags:
- `--max-turns <int>`: Maximum turns before halting (default: 100).
- `--no-thinking`: Disable thinking mode (default: thinking enabled).
- `-u, --base-url <url>`: Endpoint base URL (default: `http://192.168.100.130:4001/v1` or `$OPENAI_BASE_URL`).
- `-k, --api-key <key>`: API key (default: `sk-aisix-test` or `$OPENAI_API_KEY`).
- `-m, --model <name>`: Model ID (default: auto-detected via `/models`).

---

## Running Tests

Tests follow the **Red → Red → Green** methodology (happy path test, sad path test, implementation):

```bash
python -m unittest discover tests
```
