"""
Mem^p ReAct agent.

Implements MempAgent with:
  - LLM-based keyword extraction (memp_keyword_prompt)
  - Proceduralization: trajectory → abstract script (memp_proceduralize_prompt)
  - Adjustment: in-place script update on failure (memp_adjust_prompt)
  - Memory-augmented generator (memp_generator_prompt with {{ memory_context }})
"""

import copy
import json
import os
import re
from typing import Any

from jinja2 import Template

from appworld import AppWorld
from appworld.common.utils import read_file

from appworld_experiments.code.ace.memp_agent import ExecutionIO, MempAgent
from appworld_experiments.code.ace.memp.retriever import extract_keywords_simple
from appworld_experiments.code.ace.memp.utils import extract_json_from_text


@MempAgent.register("memp_react")
class MempReActAgent(MempAgent):
    def __init__(
        self,
        generator_prompt_file_path: str,
        proceduralize_prompt_file_path: str,
        adjust_prompt_file_path: str,
        keyword_prompt_file_path: str,
        ignore_multiple_calls: bool = True,
        max_prompt_length: int | None = None,
        max_output_length: int = 400000,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.generator_prompt_template = read_file(
            generator_prompt_file_path.replace("/", os.sep)
        ).lstrip()
        self.proceduralize_prompt_template = read_file(
            proceduralize_prompt_file_path.replace("/", os.sep)
        )
        self.adjust_prompt_template = read_file(
            adjust_prompt_file_path.replace("/", os.sep)
        )
        self.keyword_prompt_template = read_file(
            keyword_prompt_file_path.replace("/", os.sep)
        )
        self.max_prompt_length = max_prompt_length
        self.max_output_length = max_output_length
        self.ignore_multiple_calls = ignore_multiple_calls
        self.partial_code_regex = r".*```python\n(.*)"
        self.full_code_regex = r"```python\n(.*?)```"
        self.num_instruction_messages = 0

    # ------------------------------------------------------------------
    # Concept 2: keyword extraction via LLM
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
        # fallback: simple heuristic
        return extract_keywords_simple(task_desc)

    # ------------------------------------------------------------------
    # Generator prompt construction with retrieved memory context
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

    def next_execution_inputs_and_cost(
        self, last_execution_outputs: list
    ) -> tuple:
        if last_execution_outputs:
            assert len(last_execution_outputs) == 1, (
                "ReAct expects exactly one last_execution_output."
            )
            content = last_execution_outputs[0].content
            content = (
                "Output:\n```\n"
                + self._truncate_output(content)
                + "```\n\n"
            )
            self.messages.append({"role": "user", "content": content})

        output = self.generator_model.generate(messages=self.trimmed_messages)
        code, fixed_content = self._extract_code_and_fix_content(output["content"])
        self.messages.append({"role": "assistant", "content": fixed_content + "\n\n"})
        self.logger.show_message(
            role="agent", message=fixed_content, step_number=self.step_number
        )
        return [ExecutionIO(content=code)], output["cost"], None

    # ------------------------------------------------------------------
    # Concept 1: proceduralization — trajectory → abstract script
    # ------------------------------------------------------------------

    def proceduralize(self, task_desc: str, trajectory: list, success: bool) -> str:
        conv_text = "\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in trajectory
        )
        prompt = (
            self.proceduralize_prompt_template
            .replace("{{task}}", task_desc)
            .replace("{{trajectory}}", conv_text[:8000])
            .replace("{{success}}", "SUCCESS" if success else "FAILURE")
        )
        result = self.proceduralize_model.generate(
            messages=[{"role": "user", "content": prompt}]
        )
        script = result.get("content", "").strip()
        return script if script else "(No script generated)"

    # ------------------------------------------------------------------
    # Concept 3: adjustment — in-place script correction on failure
    # ------------------------------------------------------------------

    def adjust_memory(
        self, failed_task_desc: str, trajectory: list, existing_script: str
    ) -> str:
        conv_text = "\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in trajectory
        )
        prompt = (
            self.adjust_prompt_template
            .replace("{{failed_task}}", failed_task_desc)
            .replace("{{failure_trajectory}}", conv_text[:6000])
            .replace("{{existing_script}}", existing_script)
        )
        result = self.adjust_model.generate(
            messages=[{"role": "user", "content": prompt}]
        )
        updated = result.get("content", "").strip()
        return updated if updated else existing_script

    # ------------------------------------------------------------------
    # Helpers (shared with evaluation_react.py)
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

    @property
    def trimmed_messages(self) -> list:
        messages = copy.deepcopy(self.messages)
        pre = messages[: self.num_instruction_messages - 1]
        post = messages[self.num_instruction_messages - 1 :]
        text = self._messages_to_text(post)
        prefix = text[: text.index("Task: ") + 6]
        text = text.removeprefix(prefix)
        obs_idx = 0
        while len(text) > self.max_output_length:
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
