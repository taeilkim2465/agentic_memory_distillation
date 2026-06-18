import copy
import json
import os
import re
from typing import Any

from jinja2 import Template

from appworld import AppWorld
from appworld.common.utils import read_file
from appworld_experiments.code.ace.adaptation_agent import StarAgent, ExecutionIO
from .reasoning_bank import (
    load_bank,
    save_bank,
    add_memory,
    append_to_bank_safe,
    retrieve_memories,
    format_memories_for_prompt,
    parse_memory_items,
)

@StarAgent.register("ace_adaptation_react")
class SimplifiedReActStarAgent(StarAgent):
    def __init__(
        self,
        generator_prompt_file_path: str | None = None,
        reflector_prompt_file_path: str | None = None,
        reflector_success_prompt_file_path: str | None = None,
        reasoning_bank_file_path: str | None = None,
        ignore_multiple_calls: bool = True,
        max_prompt_length: int | None = None,
        max_output_length: int = 400000,
        top_k: int = 3,
        use_memory: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.generator_prompt_template = read_file(generator_prompt_file_path.replace("/", os.sep)).lstrip()
        self.reflector_prompt = read_file(reflector_prompt_file_path.replace("/", os.sep))
        if reflector_success_prompt_file_path and os.path.exists(reflector_success_prompt_file_path):
            self.reflector_success_prompt = read_file(reflector_success_prompt_file_path.replace("/", os.sep))
        else:
            self.reflector_success_prompt = self.reflector_prompt
        self.reasoning_bank_file_path = reasoning_bank_file_path
        self.max_prompt_length = max_prompt_length
        self.max_output_length = max_output_length
        self.top_k = top_k
        self.use_memory = use_memory
        self.ignore_multiple_calls = ignore_multiple_calls
        self.partial_code_regex = r".*```(?:python)?\n(.*)"
        self.full_code_regex = r"```(?:python)?\n(.*?)```"
        self.world_gt_code = None
        self.task_success = False

        self.reasoning_bank = load_bank(reasoning_bank_file_path) if reasoning_bank_file_path else []

    def initialize(self, world: AppWorld):
        super().initialize(world)
        task_instruction = world.task.instruction
        if self.use_memory:
            memories = retrieve_memories(self.reasoning_bank, query=task_instruction, top_k=self.top_k)
            memories_text = format_memories_for_prompt(memories)
        else:
            memories_text = "(No past experiences yet)"

        template = Template(self.generator_prompt_template)
        app_descriptions = json.dumps(
            [{"name": k, "description": v} for (k, v) in world.task.app_descriptions.items()],
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
        self, last_execution_outputs: list[ExecutionIO], world_gt_code: str = None, reasoning_text: str = ""
    ) -> tuple[ExecutionIO, float, str | None]:
        if world_gt_code is not None:
            self.world_gt_code = world_gt_code

        if reasoning_text != "" and reasoning_text is not None:
            self.messages.append({
                "role": "user",
                "content": "In your previous attempt, the code failed to match the ground truth outputs during unit testing. Provide reflection on what might have gone wrong and how to fix it."
            })
            self.messages.append({
                "role": "assistant",
                "content": reasoning_text + "\n\n"
            })
            self.messages.append({
                "role": "user",
                "content": "Use the reasoning above, along with the past experiences, to improve your code in all future attempts."
            })
            self.logger.show_message(role="user", message=reasoning_text, step_number=self.step_number)

        elif last_execution_outputs:
            assert (
                len(last_execution_outputs) == 1
            ), "React expects exactly one last_execution_output."
            last_execution_output_content = last_execution_outputs[0].content
            last_execution_output_content = (
                "Output:\n```\n" + self.truncate_output(last_execution_output_content) + "```\n\n"
            )
            self.messages.append({"role": "user", "content": last_execution_output_content})

        messages = self.trimmed_messages
        output = self.generator_model.generate(messages=messages)
        code, fixed_output_content = self.extract_code_and_fix_content(output["content"])
        self.messages.append({"role": "assistant", "content": fixed_output_content + "\n\n"})
        self.logger.show_message(
            role="agent", message=fixed_output_content, step_number=self.step_number
        )
        return [ExecutionIO(content=code)], output["cost"], None

    def _build_trajectory_text(self) -> str:
        lines = []
        for msg in self.trimmed_messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    def reflector_call(self, task_success: bool = False) -> str:
        prompt_template = self.reflector_success_prompt if task_success else self.reflector_prompt

        task_description = getattr(getattr(self, "world", None), "task", None)
        task_description = getattr(task_description, "instruction", "") if task_description else ""
        trajectory = self._build_trajectory_text()

        filled_prompt = (
            prompt_template
            .replace("{query}", task_description)
            .replace("{trajectory}", trajectory)
            .replace("{ground_truth_code}", self.world_gt_code or "N/A")
        )

        message_ = self.reflector_model.generate(messages=[{"role": "user", "content": filled_prompt}])
        raw_output = message_.get("content", "")

        if raw_output:
            self.logger.show_message(role="user", message=raw_output, step_number=self.step_number)
        else:
            self.logger.show_message(role="user", message="[WARN] reflector output is empty", step_number=self.step_number)

        return raw_output

    def store_to_bank(self, task_success: bool) -> None:
        if not self.use_reflector:
            return

        raw_output = self.reflector_call(task_success=task_success)
        memory_items = parse_memory_items(raw_output)

        if not memory_items:
            print("[ReasoningBank] Warning: no memory items parsed from reflector output")
            return

        task_description = getattr(getattr(self, "world", None), "task", None)
        task_description = getattr(task_description, "instruction", "") if task_description else ""
        task_id = getattr(getattr(self, "world", None), "task_id", "unknown")
        outcome = "success" if task_success else "failure"

        if self.reasoning_bank_file_path:
            self.reasoning_bank = append_to_bank_safe(
                file_path=self.reasoning_bank_file_path,
                task_id=task_id,
                task_description=task_description,
                outcome=outcome,
                memory_items=memory_items,
            )
        else:
            self.reasoning_bank = add_memory(
                self.reasoning_bank,
                task_id=task_id,
                task_description=task_description,
                outcome=outcome,
                memory_items=memory_items,
            )

    def extract_code_and_fix_content(self, text: str) -> tuple[str, str]:
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
            new_prompt = prompt[-(max_prompt_length - len(init_prompt)):]
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

    @property
    def trimmed_messages(self) -> list[dict]:
        messages = copy.deepcopy(self.messages)
        pre_messages = messages[: self.num_instruction_messages - 1]
        post_messages = messages[self.num_instruction_messages - 1:]
        output_str = self.messages_to_text(post_messages)
        remove_prefix = output_str[: output_str.index("Task: ") + 6]
        output_str = output_str.removeprefix(remove_prefix)
        observation_index = 0
        while len(output_str) > self.max_output_length:
            found_block = False
            if observation_index < len(post_messages) - 5:
                for message_index, message in enumerate(post_messages[observation_index:]):
                    if message["role"] == "user" and message["content"].startswith("Output:"):
                        message["content"] = "Output:\n```\n[NOT SHOWN FOR BREVITY]```\n\n"
                        found_block = True
                        observation_index += message_index + 1
                        break
                if not found_block:
                    observation_index = len(post_messages)
            if not found_block and len(post_messages):
                first_post_message = copy.deepcopy(post_messages[0])
                if not first_post_message["content"].endswith("[TRIMMED HISTORY]\n\n"):
                    first_post_message["content"] += "[TRIMMED HISTORY]\n\n"
                post_messages = [first_post_message] + post_messages[2:]
                found_block = True
            if not found_block:
                raise ValueError(f"No blocks found to be removed!\n{post_messages}")
            output_str = self.messages_to_text(post_messages)
            output_str = output_str.removeprefix(remove_prefix)
        messages = pre_messages + post_messages
        return messages
