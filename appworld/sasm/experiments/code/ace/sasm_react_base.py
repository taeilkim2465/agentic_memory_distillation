"""
Shared ReAct loop utilities for SASM agents.
No playbook, no reflector, no curator — pure ReAct skeleton.
"""

import copy
import re

from jinja2 import Template


class SASMReActMixin:
    """
    Mixin that provides the pure ReAct loop mechanics:
    message formatting, code extraction, input/output truncation, and message trimming.
    No reference to playbook, reflector, or curator.
    """

    partial_code_regex = r".*```python\n(.*)"
    full_code_regex = r"```python\n(.*?)```"

    # -----------------------------------------------------------------
    # Prompt initialization (called by subclass initialize())
    # -----------------------------------------------------------------

    def _setup_messages_from_template(self, template_str: str, template_params: dict) -> None:
        template = Template(template_str)
        output_str = template.render(template_params)
        output_str = self._truncate_input(output_str) + "\n\n"
        self.messages = self._text_to_messages(output_str)
        self.num_instruction_messages = len(self.messages)

    # -----------------------------------------------------------------
    # Code extraction
    # -----------------------------------------------------------------

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
        return output_code, text

    # -----------------------------------------------------------------
    # Truncation
    # -----------------------------------------------------------------

    def _truncate_input(self, input_str: str) -> str:
        if self.max_prompt_length is None:
            return input_str
        goal_index = input_str.rfind("Task:")
        if goal_index == -1:
            raise ValueError(f"No 'Task:' found in input string:\n{input_str[:200]}")
        next_nl = input_str.find("\n", goal_index) + 1
        init_prompt = input_str[:next_nl]
        prompt = input_str[next_nl:]
        if len(init_prompt) > self.max_prompt_length:
            raise ValueError("Input prompt longer than max_prompt_length")
        if len(prompt) > self.max_prompt_length - len(init_prompt):
            new_prompt = prompt[-(self.max_prompt_length - len(init_prompt)):]
            cmd_index = new_prompt.find("ASSISTANT:") if "ASSISTANT:" in new_prompt else 0
            prompt = "\n[TRIMMED HISTORY]\n\n" + new_prompt[cmd_index:]
        return init_prompt + prompt

    def _truncate_output(self, content: str) -> str:
        if len(content) > 20000:
            content = content[:20000] + "\n[REST NOT SHOWN FOR BREVITY]"
        return content

    # -----------------------------------------------------------------
    # Message serialization
    # -----------------------------------------------------------------

    def _text_to_messages(self, input_str: str) -> list[dict]:
        messages_json = []
        last_start = 0
        for m in re.finditer(r"(USER|ASSISTANT|SYSTEM):\n", input_str, flags=re.IGNORECASE):
            last_end = m.span()[0]
            if not messages_json:
                if last_end != 0:
                    raise ValueError(f"Prompt has no role at start: {input_str[:last_end]}")
            else:
                messages_json[-1]["content"] = input_str[last_start:last_end]
            messages_json.append({"role": m.group(1).lower(), "content": None})
            last_start = m.span()[1]
        messages_json[-1]["content"] = input_str[last_start:]
        return messages_json

    def _messages_to_text(self, messages: list[dict]) -> str:
        out = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                out += "SYSTEM:\n" + content
            elif role == "assistant":
                out += "ASSISTANT:\n" + content
            elif role == "user":
                out += "USER:\n" + content
        return out

    # -----------------------------------------------------------------
    # Message trimming (keeps context within max_output_length / max_model_len)
    # -----------------------------------------------------------------

    def _exceeds_context_limit(self, pre: list, post: list) -> bool:
        max_model_len = getattr(self, 'max_model_len', None)
        if max_model_len is None:
            return False
        max_output_tokens = getattr(self, 'max_output_tokens', 2048)
        context_buffer = getattr(self, 'context_buffer', 256)
        token_budget = max_model_len - max_output_tokens - context_buffer
        messages = pre + post
        tokenizer = getattr(self, '_tokenizer', None)
        if tokenizer is not None:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            count = len(tokenizer.encode(text))
        else:
            from litellm import token_counter
            lm = getattr(self, 'language_model', None) or getattr(self, 'generator_model', None)
            count = token_counter(model=getattr(lm, 'model', ''), messages=messages)
        return count > token_budget

    @property
    def trimmed_messages(self) -> list[dict]:
        messages = copy.deepcopy(self.messages)
        pre = messages[: self.num_instruction_messages - 1]
        post = messages[self.num_instruction_messages - 1:]
        output_str = self._messages_to_text(post)
        remove_prefix = output_str[: output_str.index("Task: ") + 6]
        output_str = output_str.removeprefix(remove_prefix)
        obs_idx = 0
        while self._exceeds_context_limit(pre, post) or len(output_str) > self.max_output_length:
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
                raise ValueError("No blocks to trim in messages.")
            output_str = self._messages_to_text(post).removeprefix(remove_prefix)
        return pre + post
