import os
from typing import Optional

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a bash tool for executing shell commands. "
    "When reporting results or summarizing command output, provide a concise summary without repeating items."
)


def get_prompts_dir() -> str:
    # 1. Check relative to repository root (parent directory of gemma_harness)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_prompts = os.path.join(repo_root, "prompts")
    if os.path.isdir(repo_prompts):
        return repo_prompts

    # 2. Check relative to current working directory
    cwd_prompts = os.path.join(os.getcwd(), "prompts")
    if os.path.isdir(cwd_prompts):
        return cwd_prompts

    return repo_prompts


def load_prompt(filename: str, fallback: Optional[str] = None) -> str:
    # 1. If filename is already a valid absolute or relative path, load directly
    if os.path.exists(filename) and os.path.isfile(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()

    # 2. Check within the prompts directory
    prompts_dir = get_prompts_dir()
    filepath = os.path.join(prompts_dir, filename)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()

    # 3. If missing and fallback provided, return fallback
    if fallback is not None:
        return fallback

    raise FileNotFoundError(f"Prompt file not found: {filepath} (checked {filename} and {filepath})")


def load_system_prompt(filename: str = "system_prompt.md") -> str:
    return load_prompt(filename, fallback=DEFAULT_SYSTEM_PROMPT)
