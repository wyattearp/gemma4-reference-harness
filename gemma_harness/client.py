import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Configure environment to silence PyTorch missing and unauthenticated hub warnings
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging
import huggingface_hub.utils.logging as hub_logging

hf_logging.set_verbosity_error()
hub_logging.set_verbosity_error()


class GemmaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "http://192.168.100.130:4001/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "sk-aisix-test"
        self.model_name = model_name
        self.tokenizer = None

    def discover_model(self, require_gemma: bool = False) -> str:
        if self.model_name:
            return self.model_name

        req = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch models from {self.base_url}/models: {exc}") from exc

        models_list = data.get("data", [])
        gemma_model = None
        for item in models_list:
            mid = item.get("id", "")
            if "gemma" in mid.lower():
                gemma_model = mid
                break

        if gemma_model:
            self.model_name = gemma_model
            return self.model_name

        if require_gemma:
            raise ValueError(f"No Gemma model found on {self.base_url}/models: {models_list}")

        if models_list:
            self.model_name = models_list[0].get("id")
            return self.model_name

        raise ValueError("No models returned by endpoint")

    def get_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer

        model_id = self.model_name or self.discover_model()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            # Fallback to known gemma 4 id if local tokenizer loading needs standard repo
            self.tokenizer = AutoTokenizer.from_pretrained("wyattearp/Gemma-4-26B-A4B-it-NVFP4")

        # Hot-patch response_template to natively support both <tool_call|> and <tool_response|>
        if hasattr(self.tokenizer, "response_template") and isinstance(self.tokenizer.response_template, dict):
            import copy
            patched = copy.deepcopy(self.tokenizer.response_template)
            fields = patched.get("fields", {})
            if "tool_calls" in fields and isinstance(fields["tool_calls"], dict):
                fields["tool_calls"]["close"] = ["<tool_call|>", "<tool_response|>"]
                self.tokenizer.response_template = patched

        return self.tokenizer

    def count_tokens(self, text: str) -> int:
        tok = self.get_tokenizer()
        return len(tok.encode(text, add_special_tokens=False))

    def render_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        tok = self.get_tokenizer()
        return tok.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking
        )

    def generate_completion(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        repetition_penalty: float = 1.15,
    ) -> Tuple[str, Dict[str, Any]]:
        model_id = self.model_name or self.discover_model()
        payload = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "stop": ["<turn|>", "<|turn>", "<|tool_response>", "<tool_call|>", "<tool_response|>", "<eos>"],
            "skip_special_tokens": False
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            data=body
        )
        try:
            with urllib.request.urlopen(req) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_msg = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} error from completions: {err_msg}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to communicate with completions endpoint: {exc}") from exc

        choices = resp_data.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices returned from completions endpoint: {resp_data}")

        generated_text = choices[0].get("text", "")
        return generated_text, resp_data

    def parse_output(
        self,
        generated_text: str,
        prefix: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        tok = self.get_tokenizer()
        parsed: Dict[str, Any] = {"role": "assistant"}

        # Attempt tokenizer parse_response first
        try:
            res = tok.parse_response(generated_text, prefix=prefix, tools=tools)
            if isinstance(res, dict):
                parsed.update(res)
        except Exception:
            pass

        # Fallback / safety check to guarantee thinking is captured
        if not parsed.get("thinking"):
            if "<channel|>" in generated_text:
                if prefix.endswith("<|channel>thought\n"):
                    thought_part, rest = generated_text.split("<channel|>", 1)
                    parsed["thinking"] = thought_part.strip()
                    if not parsed.get("content"):
                        parsed["content"] = rest.strip()
                else:
                    m_think = re.search(r"<\|channel>thought\n?(.*?)<channel\|>", generated_text, re.DOTALL)
                    if m_think:
                        parsed["thinking"] = m_think.group(1).strip()

        # Fallback / safety regex check to guarantee tool calls are captured
        if "tool_calls" not in parsed or not parsed["tool_calls"]:
            tool_calls = []
            pattern_loose = r"<\|tool_call>call:(\w+)\{(.*?)(?:\}(?:<tool_call\|>|<tool_response\|>|(?=<\|)|\Z)|\Z)"
            arg_regex = r'(\w+):(?:<\|"\|>(.*?)(?:<\|"\|>|$)|([^,}]*))'
            for m in re.finditer(pattern_loose, generated_text, re.DOTALL):
                name = m.group(1)
                args_str = m.group(2)
                parsed_args = {}
                for k, v_quoted, v_raw in re.findall(arg_regex, args_str, re.DOTALL):
                    if not k:
                        continue
                    val = v_quoted if v_quoted is not None and v_quoted != "" else v_raw.strip()
                    parsed_args[k] = val
                if parsed_args:
                    tool_calls.append({
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": parsed_args
                        }
                    })
            if tool_calls:
                parsed["tool_calls"] = tool_calls

        # Check if thoughts leaked into content (a known Gemma 4 edge case where an empty
        # thought channel is emitted followed by reasoning text and a second <channel|> tag)
        if "<channel|>" in parsed.get("content", ""):
            thought_leak, final_content = parsed["content"].rsplit("<channel|>", 1)
            if thought_leak.strip():
                if parsed.get("thinking"):
                    parsed["thinking"] = f"{parsed['thinking']}\n{thought_leak.strip()}".strip()
                else:
                    parsed["thinking"] = thought_leak.strip()
            parsed["content"] = final_content.strip()

        # Clean any raw tokens that might leak into content
        if parsed.get("content"):
            cleaned = parsed["content"]
            cleaned = re.sub(r"<\|channel>.*?<channel\|>", "", cleaned, flags=re.DOTALL)
            cleaned = re.sub(r"<\|tool_call>.*?<tool_call\|>", "", cleaned, flags=re.DOTALL)
            cleaned = cleaned.replace("<channel|>", "").replace("<turn|>", "").replace("<|turn>", "").replace("<|tool_response>", "").replace("<tool_response|>", "").replace("<eos>", "").strip()
            parsed["content"] = cleaned

        return parsed
