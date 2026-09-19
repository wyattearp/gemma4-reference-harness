import argparse
import os
import sys
from typing import List, Optional

from gemma_harness.agent import Agent, AgentConfig
from gemma_harness.client import GemmaClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemma-harness",
        description="Reference harness for Gemma 4 multi-turn agent with tool calling and thinking"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        help="One-shot prompt to run non-interactively"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="Maximum turns for one-shot mode (default: 100)"
    )
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=4096,
        help="Maximum tokens per model response (default: 4096)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="Repetition penalty to prevent cyclical loops (default: 1.15)"
    )
    parser.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        default=True,
        help="Enable thinking mode (default: True)"
    )
    parser.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Disable thinking mode"
    )
    parser.add_argument(
        "-u", "--base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "http://192.168.100.130:4001/v1"),
        help="Base URL for vLLM endpoint (default: http://192.168.100.130:4001/v1)"
    )
    parser.add_argument(
        "-k", "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "sk-aisix-test"),
        help="API Key for endpoint (default: sk-aisix-test)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=os.environ.get("MODEL_NAME", None),
        help="Model ID (default: auto-discover from /models)"
    )
    parser.add_argument(
        "--transcripts-dir",
        type=str,
        default=".",
        help="Directory to save transcripts (default: current directory)"
    )
    return parser


def run_cli(args_list: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_list)

    try:
        client = GemmaClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model
        )
        model_id = client.discover_model()

        config = AgentConfig(
            max_turns=args.max_turns,
            enable_thinking=args.thinking,
            max_response_tokens=args.max_response_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            transcripts_dir=args.transcripts_dir
        )

        if args.prompt:
            # One-shot mode
            def on_thought(thought: str):
                print("\n[Thinking]")
                print(thought)

            def on_tool_call(name: str, fn_args: dict):
                print(f"\n[Tool Call: {name}]")
                print(fn_args.get("command", str(fn_args)))

            def on_tool_result(name: str, result: dict):
                print(f"[Tool Result: {name} (exit: {result.get('exit_code', 0)})]")
                stdout = result.get("stdout", "").strip()
                stderr = result.get("stderr", "").strip()
                if stdout:
                    print(stdout)
                if stderr:
                    print(f"Error: {stderr}")

            agent = Agent(
                client=client,
                config=config,
                on_thought=on_thought,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result
            )

            print(f"Running task with {model_id} (thinking={'ON' if args.thinking else 'OFF'}):")
            print(f"> {args.prompt}")

            answer = agent.run_turn(args.prompt)
            print("\n[Final Answer]")
            print(answer)
            print(f"\nTranscript saved to: {agent.transcript.file_path}")
            return 0
        else:
            # Interactive Textual mode
            from gemma_harness.tui import GemmaTUI
            agent = Agent(client=client, config=config)
            app = GemmaTUI(agent)
            app.run()
            return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_cli())
