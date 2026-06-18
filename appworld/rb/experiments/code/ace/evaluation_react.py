import copy
import json
import os
import re
from typing import Any

from jinja2 import Template
import litellm
from litellm import token_counter

from appworld import AppWorld
from appworld.common.utils import read_file
from appworld_experiments.code.ace.evaluation_agent import Agent, ExecutionIO
from appworld_experiments.code.ace.reasoning_bank import (
    load_bank,
    retrieve_memories,
    format_memories_for_prompt,
)

@Agent.register("ace_evaluation_react")
class SimplifiedReActAgent(Agent):
    def __init__(
        self,
        generator_prompt_file_path: str | None = None,
        reasoning_bank_file_path: str | None = None,
        ignore_multiple_calls: bool = True,
        max_prompt_length: int | None = None,
        max_output_length: int = 400000,
        max_model_len: int | None = None,
        max_output_tokens: int = 2048,
        context_buffer: int = 256,
        top_k: int = 3,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.generator_prompt_template = read_file(generator_prompt_file_path.replace("/", os.sep)).lstrip()
        self.max_prompt_length = max_prompt_length
        self.max_output_length = max_output_length
        self.max_model_len = max_model_len
        self.max_output_tokens = max_output_tokens
        self.context_buffer = context_buffer
        self.top_k = top_k
        self.ignore_multiple_calls = ignore_multiple_calls
        self.partial_code_regex = r".*```python\n(.*)"
        self.full_code_regex = r"```python\n(.*?)```"

        if reasoning_bank_file_path and os.path.exists(reasoning_bank_file_path):
            self.reasoning_bank = load_bank(reasoning_bank_file_path)
        else:
            self.reasoning_bank = []

        self._tokenizer = None
        if self.language_model.model == "qwen3-4b":
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
                print("[Tokenizer] Loaded Qwen/Qwen3-4B tokenizer for accurate context counting.")
            except Exception as e:
                print(f"[Tokenizer] Warning: Could not load Qwen tokenizer: {e}")

    def initialize(self, world: AppWorld):
        super().initialize(world)
        task_instruction = world.task.instruction
        memories = retrieve_memories(self.reasoning_bank, query=task_instruction, top_k=self.top_k)
        memories_text = format_memories_for_prompt(memories)
        self.memories_text = memories_text

        template = Template(self.generator_prompt_template)
        app_descriptions = json.dumps(
            [{"name": name, "description": v} for (name, v) in world.task.app_descriptions.items()],
            indent=1,
        )
        template_params = {
            "input_str": world.task.instruction,
            "main_user": world.task.supervisor,
            "app_descriptions": app_descriptions,
            "relevant_apis": str(world.task.ground_truth.required_apis),
            "memories": memories_text,
        }
        output_str = template.render(template_params)
        output_str = self.truncate_input(output_str) + "\n\n"
        self.messages = self.text_to_messages(output_str)
        self.num_instruction_messages = len(self.messages)

    def next_execution_inputs_and_cost(
        self, last_execution_outputs: list[ExecutionIO], world_gt_code: str = None
    ) -> tuple[ExecutionIO, float, str | None]:
        if last_execution_outputs:
            assert (
                len(last_execution_outputs) == 1
            ), "React expects exactly one last_execution_output."
            last_execution_output_content = last_execution_outputs[0].content
            potential_new_line = ""
            last_execution_output_content = (
                "Output:\n```\n" + self.truncate_output(last_execution_output_content) + potential_new_line + "```\n\n"
            )
            self.messages.append({"role": "user", "content": last_execution_output_content})
        # Remove memories from context after the first step
        if self.step_number > 1 and self.memories_text:
            for msg in self.messages:
                if self.memories_text in msg["content"]:
                    msg["content"] = msg["content"].replace(self.memories_text, "(No past experiences)")
                    self.memories_text = None
                    break

        messages = self.trimmed_messages
        output = self.language_model.generate(messages=messages, max_tokens=self.max_output_tokens)
        code, fixed_output_content = self.extract_code_and_fix_content(output["content"])
        self.messages.append({"role": "assistant", "content": fixed_output_content + "\n\n"})
        self.logger.show_message(
            role="agent", message=fixed_output_content, step_number=self.step_number
        )
        return [ExecutionIO(content=code)], output["cost"], None

    def extract_code_and_fix_content(self, text: str) -> tuple[str, str]:
        if text is None:
            return "", ""
        original_text = text
        output_code = ""
        match_end = 0
        # Handle multiple calls
        for re_match in re.finditer(self.full_code_regex, original_text, flags=re.DOTALL):
            code = re_match.group(1).strip()
            if self.ignore_multiple_calls:
                text = original_text[: re_match.end()]
                return code, text
            output_code += code + "\n"
            match_end = re_match.end()
        # Check for partial code match at end (no terminating ```)  following the last match
        partial_match = re.match(
            self.partial_code_regex, original_text[match_end:], flags=re.DOTALL
        )
        if partial_match:
            output_code += partial_match.group(1).strip()
            # Terminated due to stop condition; add stop condition to output
            if not text.endswith("\n"):
                text = text + "\n"
            text = text + "```"
        if len(output_code) == 0:
            return "", text
        else:
            return output_code, text

    def truncate_input(self, input_str: str) -> str:
        if self.max_prompt_length is None:
            return input_str
        max_prompt_length = self.max_prompt_length
        goal_index = input_str.rfind("Task:")
        if goal_index == -1:
            raise ValueError(f"No goal found in input string:\n{input_str}")
        next_new_line_index = input_str.find("\n", goal_index) + 1
        init_prompt = input_str[:next_new_line_index]
        prompt = input_str[next_new_line_index:]
        if len(init_prompt) > max_prompt_length:
            raise ValueError("Input prompt longer than max allowed length")
        if len(prompt) > max_prompt_length - len(init_prompt):
            new_prompt = prompt[-(max_prompt_length - len(init_prompt)) :]
            cmd_index = new_prompt.find("ASSISTANT:") if "ASSISTANT:" in new_prompt else 0
            prompt = "\n[TRIMMED HISTORY]\n\n" + new_prompt[cmd_index:]
        return init_prompt + prompt
    
    def truncate_output(self, execution_output_content: str) -> str:
        if len(execution_output_content) > 20000:
            execution_output_content = execution_output_content[:20000] + "\n[REST NOT SHOWN FOR BREVITY]"
        return execution_output_content

    def text_to_messages(self, input_str: str) -> list[dict]:
        messages_json = []
        last_start = 0
        for m in re.finditer("(USER|ASSISTANT|SYSTEM):\n", input_str, flags=re.IGNORECASE):
            last_end = m.span()[0]
            if len(messages_json) == 0:
                if last_end != 0:
                    raise ValueError(
                        f"Start of the prompt has no assigned role: {input_str[:last_end]}"
                    )
            else:
                messages_json[-1]["content"] = input_str[last_start:last_end]
            role = m.group(1).lower()
            messages_json.append({"role": role, "content": None})
            last_start = m.span()[1]
        messages_json[-1]["content"] = input_str[last_start:]
        return messages_json

    def messages_to_text(self, messages: list[dict]) -> str:
        output_str = ""
        for message in messages:
            role = message["role"]
            if role == "system":
                output_str += "SYSTEM:\n" + message["content"]
            if role == "assistant":
                output_str += "ASSISTANT:\n" + message["content"]
            elif role == "user":
                output_str += "USER:\n" + message["content"]
            else:
                raise ValueError(f"Unknown message role {role} in: {message}")
        return output_str

    def _exceeds_context_limit(self, pre_messages: list[dict], post_messages: list[dict], output_str: str) -> bool:
        if self.max_model_len is not None:
            token_budget = self.max_model_len - self.max_output_tokens - self.context_buffer
            messages = pre_messages + post_messages
            if self._tokenizer is not None:
                text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                count = len(self._tokenizer.encode(text))
            else:
                count = token_counter(model=self.language_model.model, messages=messages)
            return count > token_budget
        return len(output_str) > self.max_output_length

    @property
    def trimmed_messages(self) -> list[dict]:
        messages = copy.deepcopy(self.messages)
        pre_messages = messages[: self.num_instruction_messages - 1]
        post_messages = messages[self.num_instruction_messages - 1 :]
        # post_messages[0] is the task instruction; trim oldest turns (assistant + observation pairs) from the front
        while len(post_messages) > 1 and self._exceeds_context_limit(pre_messages, post_messages, ""):
            post_messages = [post_messages[0]] + post_messages[3:]
        return pre_messages + post_messages