import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from gemma_harness.client import GemmaClient
from gemma_harness.sandbox import DockerSandbox
from gemma_harness.tools import BASH_TOOL_DECLARATION, execute_bash
from gemma_harness.transcript import TranscriptLogger


@dataclass
class AgentConfig:
    max_turns: int = 100
    enable_thinking: bool = True
    context_window: int = 256 * 1024  # 262,144 tokens
    compaction_threshold_ratio: float = 0.85  # 85% full (~222,822 tokens)
    max_response_tokens: int = 4096
    temperature: float = 0.2
    repetition_penalty: float = 1.15
    system_prompt: str = (
        "You are a helpful assistant with access to a bash tool for executing shell commands. "
        "When reporting results or summarizing command output, provide a concise summary without repeating items."
    )
    transcripts_dir: str = "."
    sandbox_mode: str = "docker"
    sandbox_image: str = "gemma4-sandbox:latest"
    workspace_dir: str = "./workspace"


class Agent:
    def __init__(
        self,
        client: GemmaClient,
        config: Optional[AgentConfig] = None,
        on_thought: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_result: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_content: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.client = client
        self.config = config or AgentConfig()
        self.on_thought = on_thought
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_content = on_content
        self.on_status = on_status

        self.history: List[Dict[str, Any]] = []
        if self.config.system_prompt:
            self.history.append({"role": "system", "content": self.config.system_prompt})

        model_name = getattr(self.client, "model_name", "gemma-4")
        self.transcript = TranscriptLogger(
            base_dir=self.config.transcripts_dir,
            model_name=model_name or "gemma-4"
        )
        self.tools = [BASH_TOOL_DECLARATION]

        self.sandbox: Optional[DockerSandbox] = None
        if self.config.sandbox_mode == "docker":
            self.sandbox = DockerSandbox(
                image=self.config.sandbox_image,
                workspace_dir=self.config.workspace_dir,
            )

    @property
    def compaction_threshold(self) -> int:
        return int(self.config.context_window * self.config.compaction_threshold_ratio)

    def count_context_tokens(self) -> int:
        try:
            rendered = self.client.render_prompt(
                self.history,
                tools=self.tools,
                enable_thinking=self.config.enable_thinking,
                add_generation_prompt=False
            )
            return self.client.count_tokens(rendered)
        except Exception:
            # Fallback estimation: ~4 chars per token
            chars = sum(len(str(m.get("content", ""))) for m in self.history)
            return chars // 4

    def get_context_status(self) -> str:
        current_tokens = self.count_context_tokens()
        threshold = self.compaction_threshold
        left = max(0, threshold - current_tokens)
        pct = (current_tokens / self.config.context_window) * 100
        return f"{left:,} tokens left before compacting ({pct:.1f}% of 256K used)"

    def maybe_compact(self) -> bool:
        current_tokens = self.count_context_tokens()
        threshold = self.compaction_threshold
        if current_tokens < threshold:
            return False

        old_token_count = current_tokens
        old_len = len(self.history)

        # Sliding window compaction:
        # Keep system prompt (index 0 if system/developer), and keep recent turns.
        has_system = len(self.history) > 0 and self.history[0].get("role") in ("system", "developer")
        start_idx = 1 if has_system else 0

        # Discard oldest turns until we are well below threshold (e.g. at 60% or keep last 6 turns)
        target_tokens = int(self.config.context_window * 0.60)
        while len(self.history) > start_idx + 2:
            # Drop the oldest user+assistant turn
            self.history.pop(start_idx)
            if len(self.history) > start_idx and self.history[start_idx].get("role") == "assistant":
                self.history.pop(start_idx)

            cur = self.count_context_tokens()
            if cur <= target_tokens:
                break

        new_token_count = self.count_context_tokens()
        removed_turns = old_len - len(self.history)
        self.transcript.log_compaction(old_token_count, new_token_count, removed_turns)
        if self.on_status:
            self.on_status(f"Compacted {removed_turns} messages. {self.get_context_status()}")
        return True

    def clear(self):
        self.history = []
        if self.config.system_prompt:
            self.history.append({"role": "system", "content": self.config.system_prompt})
        model_name = getattr(self.client, "model_name", "gemma-4")
        self.transcript = TranscriptLogger(
            base_dir=self.config.transcripts_dir,
            model_name=model_name or "gemma-4"
        )
        if self.on_status:
            self.on_status(f"Session cleared. {self.get_context_status()}")

    def extract_command(self, fn_args: Any) -> str:
        if isinstance(fn_args, str):
            raw_cmd = fn_args
        elif isinstance(fn_args, dict):
            if "command" in fn_args:
                raw_cmd = str(fn_args["command"])
            elif "1" in fn_args:
                raw_cmd = str(fn_args["1"])
            elif "0" in fn_args:
                raw_cmd = str(fn_args["0"])
            elif len(fn_args) == 1:
                raw_cmd = str(next(iter(fn_args.values())))
            else:
                raw_cmd = ""
        else:
            raw_cmd = ""

        raw_cmd = raw_cmd.strip()
        if "```" in raw_cmd:
            match = re.search(r"```(?:bash|sh)?\n?(.*?)\n?```", raw_cmd, re.DOTALL)
            if match:
                return match.group(1).strip()

        return raw_cmd

    def run_turn(self, user_input: str) -> str:
        # 1. Add user message to history
        self.history.append({"role": "user", "content": user_input})
        self.maybe_compact()

        turn_count = 0
        final_answer = ""
        assistant_turn: Optional[Dict[str, Any]] = None

        while turn_count < self.config.max_turns:
            turn_count += 1

            # Prepare prompt
            if assistant_turn is None:
                messages_to_render = self.history
            else:
                messages_to_render = self.history + [assistant_turn]

            prompt = self.client.render_prompt(
                messages_to_render,
                tools=self.tools,
                enable_thinking=self.config.enable_thinking,
                add_generation_prompt=(assistant_turn is None)
            )
            prompt_tokens = self.client.count_tokens(prompt)
            self.transcript.log_prompt(prompt, token_count=prompt_tokens)

            # Generate completion
            raw_text, raw_data = self.client.generate_completion(
                prompt,
                max_tokens=self.config.max_response_tokens,
                temperature=self.config.temperature,
                repetition_penalty=self.config.repetition_penalty
            )
            raw_tokens = self.client.count_tokens(raw_text)
            finish_reason = None
            if isinstance(raw_data, dict):
                choices = raw_data.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], dict):
                    finish_reason = choices[0].get("finish_reason")
            self.transcript.log_raw_completion(raw_text, token_count=raw_tokens, stop_reason=finish_reason)

            # Parse completion
            parsed = self.client.parse_output(raw_text, prefix=prompt, tools=self.tools)
            self.transcript.log_parsed_turn(parsed)

            # Record thoughts if present
            thinking = parsed.get("thinking")
            if thinking and self.on_thought:
                self.on_thought(thinking)

            # Check for tool calls
            tool_calls = parsed.get("tool_calls", [])
            if tool_calls:
                tool_responses = []

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    fn_args = fn.get("arguments", {})
                    if self.on_tool_call:
                        self.on_tool_call(fn_name, fn_args)

                    if finish_reason == "length" and not (raw_text.strip().endswith("<tool_call|>") or raw_text.strip().endswith("<tool_response|>")):
                        result = {
                            "exit_code": 1,
                            "stdout": "",
                            "stderr": "Error: Tool execution aborted because model generation reached the maximum token limit before completing the tool call."
                        }
                    elif fn_name == "bash":
                        cmd = self.extract_command(fn_args)
                        if self.sandbox:
                            result = self.sandbox.execute(cmd)
                        else:
                            result = execute_bash(cmd)
                    else:
                        result = {"exit_code": 1, "stdout": "", "stderr": f"Unknown tool: {fn_name}"}

                    self.transcript.log_tool_call(fn_name, fn_args, result)
                    if self.on_tool_result:
                        self.on_tool_result(fn_name, result)

                    tool_responses.append({
                        "name": fn_name,
                        "response": result
                    })

                assistant_turn = {
                    "role": "assistant",
                    "reasoning": thinking or "",
                    "tool_calls": tool_calls,
                    "tool_responses": tool_responses
                }
                # Tool executed, loop forward to let model process tool response
                continue

            # No tool calls: final answer produced
            content = parsed.get("content", "")
            if assistant_turn is not None:
                assistant_turn["content"] = content
                if thinking and not assistant_turn.get("reasoning"):
                    assistant_turn["reasoning"] = thinking
                self.history.append(assistant_turn)
            else:
                self.history.append({
                    "role": "assistant",
                    "reasoning": thinking or "",
                    "content": content
                })

            final_answer = content
            if self.on_content:
                self.on_content(content)
            break
        else:
            raise RuntimeError(f"Exceeded maximum turns ({self.config.max_turns}) without completing task")

        if self.on_status:
            self.on_status(self.get_context_status())

        return final_answer
