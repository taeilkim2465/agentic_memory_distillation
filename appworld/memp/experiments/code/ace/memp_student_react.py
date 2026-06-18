"""
Mem^p Student ReAct agent.

Reads from teacher's memory store (AveFact retrieval) and solves tasks.
Never writes to the memory store.
"""

import copy
import json
import os
import re
from typing import Any

from jinja2 import Template

from appworld import AppWorld
from appworld.common.utils import read_file

from appworld_experiments.code.ace.memp_student_agent import ExecutionIO, MempStudentAgent
from appworld_experiments.code.ace.memp.retriever import extract_keywords_simple
from appworld_experiments.code.ace.memp.utils import extract_json_from_text


@MempStudentAgent.register("memp_student_react")
class MempStudentReActAgent(MempStudentAgent):
    def __init__(
        self,
        generator_prompt_file_path: str,
        keyword_prompt_file_path: str,
        ignore_multiple_calls: bool = True,
        max_prompt_length: int | None = None,
        max_output_length: int = 400000,
        max_model_len: int | None = None,
        max_output_tokens: int = 2048,
        context_buffer: int = 256,
        max_consecutive_failures: int = 3,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.generator_prompt_template = read_file(
            generator_prompt_file_path.replace("/", os.sep)
        ).lstrip()
        self.keyword_prompt_template = read_file(
            keyword_prompt_file_path.replace("/", os.sep)
        )
        self.max_prompt_length = max_prompt_length
        self.max_output_length = max_output_length
        self.max_model_len = max_model_len
        self.max_output_tokens = max_output_tokens
        self.context_buffer = context_buffer
        self.ignore_multiple_calls = ignore_multiple_calls
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self._recent_codes: list[str] = []
        self._recent_error_sigs: list[str] = []
        self._tokenizer = None
        if self.generator_model.model == "qwen3-4b":
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
                print("[Tokenizer] Loaded Qwen/Qwen3-4B tokenizer for accurate context counting.")
            except Exception as e:
                print(f"[Tokenizer] Warning: Could not load Qwen tokenizer: {e}")
        self.partial_code_regex = r".*```python\n(.*)"
        self.full_code_regex = r"```python\n(.*?)```"
        self.num_instruction_messages = 0

    # ------------------------------------------------------------------
    # Keyword extraction (same LLM call as teacher)
    # ------------------------------------------------------------------

    def extract_keywords(self, task_desc: str) -> list:
        prompt = self.keyword_prompt_template.replace("{{task}}", task_desc)
        result = self.keyword_model.generate(
            messages=[{"role": "user", "content": prompt}]
        )
        raw = result.get("content", "")
        parsed = extract_json_from_text(raw)
        if parsed and isinstance(parsed.get("keywords"), list):
            return [str(k) for k in parsed["keywords"]]
        return extract_keywords_simple(task_desc)

    # ------------------------------------------------------------------
    # Per-task initialization
    # ------------------------------------------------------------------

    def initialize(self, world: AppWorld):
        super().initialize(world)
        self._consecutive_failures = 0
        self._recent_codes = []
        self._recent_error_sigs = []

    # ------------------------------------------------------------------
    # Prompt construction using retrieved teacher memory
    # ------------------------------------------------------------------

    def build_prompt(self, world: AppWorld) -> None:
        template = Template(self.generator_prompt_template)
        app_descriptions = json.dumps(
            [
                {"name": k, "description": v}
                for k, v in world.task.app_descriptions.items()
            ],
            indent=1,
        )
        memory_context = self.build_memory_context(self.retrieved_memories)
        template_params = {
            "input_str": world.task.instruction,
            "main_user": world.task.supervisor,
            "app_descriptions": app_descriptions,
            "relevant_apis": str(world.task.ground_truth.required_apis),
            "memory_context": memory_context,
        }
        output_str = template.render(template_params)
        output_str = self._truncate_input(output_str) + "\n\n"
        self.messages = self._text_to_messages(output_str)
        self.num_instruction_messages = len(self.messages)

    # ------------------------------------------------------------------
    # ReAct execution loop
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_error_sig(output: str) -> str | None:
        """에러 출력에서 재현 가능한 짧은 시그니처를 추출한다."""
        if "Execution failed." not in output and "Traceback" not in output:
            return None
        m = re.search(r'(status code is \d+.*?)(?:\n|$)', output)
        if m:
            return m.group(1)[:120]
        for line in output.splitlines():
            if "Exception" in line or "Error" in line:
                return line.strip()[:120]
        return output[:120]

    def _build_loop_hint(self, raw: str, code: str) -> str:
        """반복/루프 감지 후 적절한 힌트 문자열을 반환한다. 없으면 빈 문자열."""
        K = self.max_consecutive_failures

        # 동일 코드 반복 감지
        same_code = (
            len(self._recent_codes) >= K
            and len(set(self._recent_codes[-K:])) == 1
        )
        # 동일 에러 반복 감지
        sig = self._extract_error_sig(raw)
        if sig:
            self._recent_error_sigs.append(sig)
        same_error = (
            sig is not None
            and len(self._recent_error_sigs) >= K
            and len(set(self._recent_error_sigs[-K:])) == 1
        )

        if same_code and same_error:
            return (
                f"🚨 CRITICAL: You have run the exact same code {K} times and received "
                f"the same error every time: \"{sig}\". "
                "This approach is fundamentally broken. "
                "You MUST try a completely different method — different API, different parameters, "
                "or re-read the API docs to find the correct approach.\n\n"
            )
        if same_code:
            return (
                f"⚠️ You have executed the exact same code {K} times in a row. "
                "Stop and try a different approach.\n\n"
            )
        if same_error:
            return (
                f"⚠️ The same error has occurred {K} times: \"{sig}\". "
                "This specific error requires a different strategy — "
                "check the API docs or use a different parameter.\n\n"
            )
        if self._consecutive_failures >= K:
            severity = min(self._consecutive_failures // K, 2)
            if severity >= 2:
                return (
                    "🚨 You have been failing repeatedly. Stop all current attempts. "
                    "Re-read the task from scratch and choose an entirely different approach.\n\n"
                )
            return (
                "⚠️ You have encountered execution errors multiple consecutive times. "
                "Stop repeating the same approach. Think carefully about why it is failing "
                "and try a completely different strategy.\n\n"
            )
        return ""

    def next_execution_inputs_and_cost(self, last_execution_outputs: list) -> tuple:
        if last_execution_outputs:
            assert len(last_execution_outputs) == 1
            raw = last_execution_outputs[0].content
            if "Execution failed." in raw or "Traceback" in raw:
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0
                self._recent_error_sigs.clear()
            content = "Output:\n```\n" + self._truncate_output(raw) + "```\n\n"
            self.messages.append({"role": "user", "content": content})

        # 루프 힌트: generate 전에 주입 (에이전트가 보고 판단하도록)
        hint = ""
        if last_execution_outputs:
            hint = self._build_loop_hint(
                last_execution_outputs[0].content,
                self._recent_codes[-1] if self._recent_codes else "",
            )
        if hint:
            self.messages.append({"role": "user", "content": hint})

        try:
            output = self.generator_model.generate(messages=self.trimmed_messages, max_tokens=self.max_output_tokens)
        except Exception as e:
            err = str(e)
            m = re.search(r"prompt contains at least (\d+) input tokens", err)
            if m and self.max_model_len is not None:
                actual = int(m.group(1))
                fallback_max_tokens = max(64, self.max_model_len - actual - 100)
                print(f"[context] 400 error: prompt={actual} tokens, retrying with max_tokens={fallback_max_tokens}")
                output = self.generator_model.generate(messages=self.trimmed_messages, max_tokens=fallback_max_tokens)
            else:
                raise
        code, fixed_content = self._extract_code_and_fix_content(output["content"])
        self._recent_codes.append(code.strip())
        self.messages.append({"role": "assistant", "content": fixed_content + "\n\n"})
        self.logger.show_message(
            role="agent", message=fixed_content, step_number=self.step_number
        )
        return [ExecutionIO(content=code)], output["cost"], None

    # ------------------------------------------------------------------
    # Helpers (identical to memp_react.py)
    # ------------------------------------------------------------------

    def _extract_code_and_fix_content(self, text: str) -> tuple:
        if text is None:
            return "", ""
        original_text = text
        output_code = ""
        match_end = 0
        for re_match in re.finditer(self.full_code_regex, original_text, flags=re.DOTALL):
            code = re_match.group(1).strip()
            if self.ignore_multiple_calls:
                text = original_text[: re_match.end()]
                return code, text
            output_code += code + "\n"
            match_end = re_match.end()
        partial_match = re.match(
            self.partial_code_regex, original_text[match_end:], flags=re.DOTALL
        )
        if partial_match:
            output_code += partial_match.group(1).strip()
            if not text.endswith("\n"):
                text = text + "\n"
            text = text + "```"
        return ("", text) if not output_code else (output_code, text)

    def _truncate_input(self, input_str: str) -> str:
        if self.max_prompt_length is None:
            return input_str
        goal_index = input_str.rfind("Task:")
        if goal_index == -1:
            raise ValueError(f"No 'Task:' found in input:\n{input_str[:200]}")
        next_nl = input_str.find("\n", goal_index) + 1
        init_prompt = input_str[:next_nl]
        prompt = input_str[next_nl:]
        if len(init_prompt) > self.max_prompt_length:
            raise ValueError("Init prompt longer than max_prompt_length")
        budget = self.max_prompt_length - len(init_prompt)
        if len(prompt) > budget:
            tail = prompt[-budget:]
            cmd_idx = tail.find("ASSISTANT:") if "ASSISTANT:" in tail else 0
            prompt = "\n[TRIMMED HISTORY]\n\n" + tail[cmd_idx:]
        return init_prompt + prompt

    def _truncate_output(self, content: str) -> str:
        if len(content) > 20000:
            content = content[:20000] + "\n[REST NOT SHOWN FOR BREVITY]"
        return content

    def _text_to_messages(self, input_str: str) -> list:
        messages = []
        last_start = 0
        for m in re.finditer(r"(USER|ASSISTANT|SYSTEM):\n", input_str, flags=re.IGNORECASE):
            last_end = m.span()[0]
            if not messages:
                if last_end != 0:
                    raise ValueError(
                        f"Prompt has no role at start: {input_str[:last_end]}"
                    )
            else:
                messages[-1]["content"] = input_str[last_start:last_end]
            messages.append({"role": m.group(1).lower(), "content": None})
            last_start = m.span()[1]
        messages[-1]["content"] = input_str[last_start:]
        return messages

    def _messages_to_text(self, messages: list) -> str:
        out = ""
        for msg in messages:
            role = msg["role"]
            if role == "system":
                out += "SYSTEM:\n" + msg["content"]
            elif role == "assistant":
                out += "ASSISTANT:\n" + msg["content"]
            elif role == "user":
                out += "USER:\n" + msg["content"]
        return out

    def _exceeds_context_limit(self, pre: list, post: list) -> bool:
        if self.max_model_len is not None:
            token_budget = self.max_model_len - self.max_output_tokens - self.context_buffer
            messages = pre + post
            if self._tokenizer is not None:
                text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                count = len(self._tokenizer.encode(text))
            else:
                from litellm import token_counter
                count = token_counter(model=self.generator_model.model, messages=messages)
            return count > token_budget
        return False

    @property
    def trimmed_messages(self) -> list:
        messages = copy.deepcopy(self.messages)
        pre = messages[: self.num_instruction_messages - 1]
        post = messages[self.num_instruction_messages - 1 :]
        text = self._messages_to_text(post)
        prefix = text[: text.index("Task: ") + 6]
        text = text.removeprefix(prefix)
        obs_idx = 0
        while self._exceeds_context_limit(pre, post) or len(text) > self.max_output_length:
            found = False
            if obs_idx < len(post) - 5:
                for mi, msg in enumerate(post[obs_idx:]):
                    if msg["role"] == "user" and msg["content"].startswith("Output:"):
                        msg["content"] = "Output:\n```\n[NOT SHOWN FOR BREVITY]```\n\n"
                        found = True
                        obs_idx += mi + 1
                        break
                if not found:
                    obs_idx = len(post)
            if not found and post:
                first = copy.deepcopy(post[0])
                if not first["content"].endswith("[TRIMMED HISTORY]\n\n"):
                    first["content"] += "[TRIMMED HISTORY]\n\n"
                post = [first] + post[2:]
                found = True
            if not found:
                raise ValueError(f"No blocks to remove!\n{post}")
            text = self._messages_to_text(post)
            text = text.removeprefix(prefix)
        return pre + post
