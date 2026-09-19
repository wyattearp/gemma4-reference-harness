import json
import os
from datetime import datetime


def default_json_serializer(obj):
    return str(obj)


class TranscriptLogger:
    def __init__(self, base_dir: str = ".", model_name: str = "unknown"):
        self.base_dir = base_dir
        self.model_name = model_name
        self.start_time = datetime.now()
        self.isodate = self.start_time.strftime("%Y-%m-%dT%H-%M-%S")
        self.events = []

        self.transcripts_dir = os.path.join(self.base_dir, "transcripts")
        os.makedirs(self.transcripts_dir, exist_ok=True)

        self.symlink_dir = os.path.join(self.base_dir, "transcripts")
        if not os.path.exists(self.symlink_dir):
            try:
                os.symlink("transcripts", self.symlink_dir)
            except OSError:
                pass

        self.file_path = os.path.join(self.transcripts_dir, f"{self.isodate}.jsonc")

    def log_prompt(self, prompt: str, token_count: int = 0):
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "prompt_sent",
            "token_count": token_count,
            "prompt": prompt
        })
        self.save()

    def log_raw_completion(self, raw_text: str, token_count: int = 0, stop_reason: str = None):
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "raw_completion_received",
            "token_count": token_count,
            "stop_reason": stop_reason,
            "text": raw_text
        })
        self.save()

    def log_parsed_turn(self, parsed: dict):
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "parsed_turn",
            "parsed": parsed
        })
        self.save()

    def log_tool_call(self, tool_name: str, arguments: dict, result: dict):
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "tool_execution",
            "tool": tool_name,
            "arguments": arguments,
            "result": result
        })
        self.save()

    def log_compaction(self, old_token_count: int, new_token_count: int, removed_turns: int):
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "compaction_event",
            "old_token_count": old_token_count,
            "new_token_count": new_token_count,
            "removed_turns": removed_turns
        })
        self.save()

    def save(self) -> str:
        data = {
            "session_start": self.start_time.isoformat(),
            "model": self.model_name,
            "events_count": len(self.events),
            "events": self.events
        }
        json_body = json.dumps(data, indent=2, default=default_json_serializer)
        content = (
            f"// Gemma 4 Reference Harness Transcript\n"
            f"// Session: {self.isodate} | Model: {self.model_name}\n"
            f"{json_body}\n"
        )
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return self.file_path
