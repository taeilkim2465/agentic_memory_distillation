"""
SASM Student Agent (Evaluation Phase).

Two variants:
- sasm_evaluation_react: dynamic injection at subtask transitions (agent self-reports via XML tags)
- sasm_upfront_evaluation_react: upfront injection — student model decomposes task first,
    classifies into z categories, retrieves matching experiences, injects all before ReAct loop.
"""

import json
import os
import re
from typing import Any

from appworld import AppWorld
from appworld.common.utils import read_file
from appworld_experiments.code.ace.evaluation_agent import Agent, ExecutionIO
from appworld_experiments.code.ace.playbook import extract_json_from_text
from appworld_experiments.code.ace.sasm_memory import SASMMemoryBank, CATEGORIES
from appworld_experiments.code.ace.sasm_react_base import SASMReActMixin

TRANSITION_RE = re.compile(
    r'<subtask_transition>(.*?)</subtask_transition>',
    re.DOTALL | re.IGNORECASE,
)


@Agent.register("sasm_evaluation_react")
class SASMEvaluationReActAgent(SASMReActMixin, Agent):
    """
    Student agent that uses SASM memories during inference.
    Inherits the solve_task/solve_tasks loop from Agent.
    Uses SASMReActMixin for ReAct mechanics (no playbook).
    """

    EXPERIENCE_HEADER = "[SASM Retrieved Experience]\nCategory: {z}\n{e}\n[End of SASM Experience]\n\n"

    def __init__(
        self,
        generator_model_config: dict,
        generator_prompt_file_path: str,
        sasm_memory_file_path: str,
        sasm_predictor_prompt_file_path: str | None = None,  # unused; kept for config compatibility
        predictor_model_config: dict | None = None,           # unused; kept for config compatibility
        ignore_multiple_calls: bool = True,
        max_prompt_length: int | None = None,
        max_output_length: int = 400000,
        max_model_len: int | None = None,
        max_output_tokens: int = 2048,
        context_buffer: int = 256,
        max_consecutive_failures: int = 3,
        **kwargs: Any,
    ):
        super().__init__(generator_model_config=generator_model_config, **kwargs)
        self.generator_prompt_template = read_file(
            generator_prompt_file_path.replace("/", os.sep)
        ).lstrip()
        self.memory_bank = SASMMemoryBank(sasm_memory_file_path)
        self.ignore_multiple_calls = ignore_multiple_calls
        self.max_prompt_length = max_prompt_length
        self.max_output_length = max_output_length
        self.max_model_len = max_model_len
        self.max_output_tokens = max_output_tokens
        self.context_buffer = context_buffer
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self._recent_codes: list[str] = []
        self._current_z: str | None = None
        self._experience_in_messages: bool = False
        self._tokenizer = None
        if self.language_model.model == "qwen3-4b":
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
                print("[Tokenizer] Loaded Qwen/Qwen3-4B tokenizer for accurate context counting.")
            except Exception as e:
                print(f"[Tokenizer] Warning: Could not load Qwen tokenizer: {e}")
        self._current_task_instruction: str = ""

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def initialize(self, world: AppWorld) -> None:
        super().initialize(world)
        self._consecutive_failures = 0
        self._recent_codes = []
        self._current_z = None
        self._experience_in_messages = False
        self._current_task_instruction = getattr(world.task, "instruction", "")
        app_descriptions = json.dumps(
            [{"name": k, "description": v}
             for k, v in world.task.app_descriptions.items()],
            indent=1,
        )
        self._setup_messages_from_template(
            self.generator_prompt_template,
            {
                "input_str": world.task.instruction,
                "main_user": world.task.supervisor,
                "app_descriptions": app_descriptions,
            },
        )
        # 모든 태스크는 explore subtask로 시작
        self._transition_to("explore", self._current_task_instruction[:100])

    # ------------------------------------------------------------------
    # Subtask transition helpers
    # ------------------------------------------------------------------

    def _parse_transition_signal(self, thought: str) -> tuple[str, str] | None:
        """에이전트 thought에서 subtask 전환 신호를 파싱한다."""
        match = TRANSITION_RE.search(thought)
        if not match:
            return None
        parsed = extract_json_from_text(match.group(1).strip())
        if not parsed:
            return None
        next_z = parsed.get("next_z", "")
        next_d = parsed.get("next_d", self._current_task_instruction[:100])
        if next_z not in CATEGORIES:
            return None
        return next_z, next_d

    def _transition_to(self, next_z: str, next_d: str) -> None:
        """기존 experience를 제거하고 next_z의 experience를 주입한다."""
        if next_z == self._current_z:
            print(f"[SASM] same category '{next_z}' — skipping re-injection")
            return
        if self._experience_in_messages:
            self._remove_experience_from_messages()
            self._experience_in_messages = False

        experience = self.memory_bank.retrieve(next_z, next_d)
        if experience:
            exp_msg = self.EXPERIENCE_HEADER.format(z=next_z, e=experience)
            self.messages.append({"role": "user", "content": exp_msg})
            self._experience_in_messages = True
            print(f"[SASM] {self._current_z} → {next_z}: experience 주입")
        else:
            print(f"[SASM] {self._current_z} → {next_z}: experience 없음")
        self._current_z = next_z

    # ------------------------------------------------------------------
    # Memory-augmented ReAct step
    # ------------------------------------------------------------------

    def next_execution_inputs_and_cost(
        self,
        last_execution_outputs: list[ExecutionIO],
        world_gt_code: str = None,
    ) -> tuple[list[ExecutionIO], float, str | None]:
        # Append last execution output
        if last_execution_outputs:
            assert len(last_execution_outputs) == 1
            raw = last_execution_outputs[0].content
            out_content = (
                "Output:\n```\n"
                + self._truncate_output(raw)
                + "```\n\n"
            )
            self.messages.append({"role": "user", "content": out_content})

        # Detect repeated code and inject hint
        is_repeated = (
            len(self._recent_codes) >= self.max_consecutive_failures
            and len(set(self._recent_codes[-self.max_consecutive_failures:])) == 1
        )
        if is_repeated:
            hint = (
                "⚠️ You have been executing the exact same code repeatedly without making progress. "
                "Stop and try a completely different approach.\n\n"
            )
            self.messages.append({"role": "user", "content": hint})

        # Generate (current subtask experience already in context)
        try:
            output = self.language_model.generate(messages=self.trimmed_messages, max_tokens=self.max_output_tokens)
        except Exception as e:
            err = str(e)
            m = re.search(r"prompt contains at least (\d+) input tokens", err)
            if m and self.max_model_len is not None:
                actual = int(m.group(1))
                fallback_max_tokens = max(64, self.max_model_len - actual - 100)
                print(f"[context] 400 error: prompt={actual} tokens, retrying with max_tokens={fallback_max_tokens}")
                output = self.language_model.generate(messages=self.trimmed_messages, max_tokens=fallback_max_tokens)
            else:
                raise
        code, fixed_content = self.extract_code_and_fix_content(output["content"])
        self._recent_codes.append(code.strip())
        self.messages.append({"role": "assistant", "content": fixed_content + "\n\n"})
        self.logger.show_message(
            role="agent", message=fixed_content, step_number=self.step_number
        )

        # 에이전트 thought에서 subtask 전환 신호 파싱
        try:
            transition = self._parse_transition_signal(fixed_content)
            if transition:
                next_z, next_d = transition
                self._transition_to(next_z, next_d)
        except Exception as exc:
            print(f"[SASM] Transition parsing failed: {exc}")

        return [ExecutionIO(content=code)], output["cost"], None

    def _remove_experience_from_messages(self) -> None:
        for i, msg in enumerate(self.messages):
            if msg["content"].startswith("[SASM Retrieved Experience]"):
                self.messages.pop(i)
                return


@Agent.register("sasm_upfront_evaluation_react")
class SASMUpfrontEvaluationReActAgent(SASMReActMixin, Agent):
    """
    Student agent that decomposes the task upfront before the ReAct loop.
    Steps:
      1. Call student LLM with task instruction → list of predicted (z, d) subtasks
      2. For each unique z, retrieve the best matching experience from memory bank
      3. Inject all retrieved experiences as a single block into messages
      4. Run normal ReAct loop (no further injection)
    """

    EXPERIENCE_ITEM = "[{z} phase]\n{e}\n"

    def __init__(
        self,
        generator_model_config: dict,
        generator_prompt_file_path: str,
        sasm_memory_file_path: str,
        upfront_decompose_prompt_file_path: str,
        sasm_predictor_prompt_file_path: str | None = None,
        predictor_model_config: dict | None = None,
        ignore_multiple_calls: bool = True,
        max_prompt_length: int | None = None,
        max_output_length: int = 400000,
        max_model_len: int | None = None,
        max_output_tokens: int = 2048,
        context_buffer: int = 256,
        max_consecutive_failures: int = 3,
        **kwargs: Any,
    ):
        super().__init__(generator_model_config=generator_model_config, **kwargs)
        self.generator_prompt_template = read_file(
            generator_prompt_file_path.replace("/", os.sep)
        ).lstrip()
        self.upfront_decompose_prompt = read_file(
            upfront_decompose_prompt_file_path.replace("/", os.sep)
        ).strip()
        self.memory_bank = SASMMemoryBank(sasm_memory_file_path)
        self.ignore_multiple_calls = ignore_multiple_calls
        self.max_prompt_length = max_prompt_length
        self.max_output_length = max_output_length
        self.max_model_len = max_model_len
        self.max_output_tokens = max_output_tokens
        self.context_buffer = context_buffer
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self._recent_codes: list[str] = []
        self._tokenizer = None
        if self.language_model.model == "qwen3-4b":
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
                print("[Tokenizer] Loaded Qwen/Qwen3-4B tokenizer for accurate context counting.")
            except Exception as e:
                print(f"[Tokenizer] Warning: Could not load Qwen tokenizer: {e}")

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def initialize(self, world: AppWorld) -> None:
        super().initialize(world)
        self._consecutive_failures = 0
        self._recent_codes = []
        task_instruction = getattr(world.task, "instruction", "")
        app_descriptions = json.dumps(
            [{"name": k, "description": v}
             for k, v in world.task.app_descriptions.items()],
            indent=1,
        )
        # Build experiences first so they can be injected inside the template
        experiences_text = self._build_experiences_text(task_instruction)
        self._setup_messages_from_template(
            self.generator_prompt_template,
            {
                "input_str": task_instruction,
                "main_user": world.task.supervisor,
                "app_descriptions": app_descriptions,
                "sasm_experiences": experiences_text,
            },
        )

    # ------------------------------------------------------------------
    # Upfront decompose + retrieve
    # ------------------------------------------------------------------

    def _decompose_task(self, task_instruction: str) -> list[dict]:
        """Call student LLM to predict (z, d) subtask sequence from task instruction."""
        prompt = self.upfront_decompose_prompt.replace("{{task_instruction}}", task_instruction)
        messages = [{"role": "user", "content": prompt}]
        try:
            result = self.language_model.generate(messages=messages, max_tokens=512)
            raw = result["content"].strip()
            parsed = extract_json_from_text(raw)
            if parsed and "subtasks" in parsed:
                subtasks = [
                    st for st in parsed["subtasks"]
                    if isinstance(st, dict) and st.get("z") in CATEGORIES
                ]
                print(f"[SASM Upfront] Decomposed into {len(subtasks)} subtasks: "
                      f"{[st['z'] for st in subtasks]}")
                return subtasks
        except Exception as e:
            print(f"[SASM Upfront] Decompose failed: {e}")
        # Fallback: standard AppWorld task flow
        return [
            {"z": "explore",      "d": task_instruction[:80]},
            {"z": "authenticate", "d": task_instruction[:80]},
            {"z": "query",        "d": task_instruction[:80]},
            {"z": "execute",      "d": task_instruction[:80]},
            {"z": "verify",       "d": task_instruction[:80]},
        ]

    def _build_experiences_text(self, task_instruction: str) -> str:
        """Decompose task, retrieve one experience per z category, return as formatted string."""
        subtasks = self._decompose_task(task_instruction)

        seen: set[str] = set()
        items: list[str] = []
        for st in subtasks:
            z, d = st["z"], st.get("d", task_instruction[:80])
            if z in seen:
                continue
            seen.add(z)
            exp = self.memory_bank.retrieve(z, d)
            if exp:
                items.append(self.EXPERIENCE_ITEM.format(z=z, e=exp))
                print(f"[SASM Upfront] Retrieved experience for z='{z}'")
            else:
                print(f"[SASM Upfront] No experience found for z='{z}'")

        if items:
            print(f"[SASM Upfront] Built {len(items)} experience(s) for template injection.")
            return "\n".join(items)
        print("[SASM Upfront] No experiences found.")
        return ""

    # ------------------------------------------------------------------
    # ReAct step (no dynamic injection)
    # ------------------------------------------------------------------

    def next_execution_inputs_and_cost(
        self,
        last_execution_outputs: list[ExecutionIO],
        world_gt_code: str = None,
    ) -> tuple[list[ExecutionIO], float, str | None]:
        if last_execution_outputs:
            assert len(last_execution_outputs) == 1
            raw = last_execution_outputs[0].content
            out_content = (
                "Output:\n```\n"
                + self._truncate_output(raw)
                + "```\n\n"
            )
            self.messages.append({"role": "user", "content": out_content})

        is_repeated = (
            len(self._recent_codes) >= self.max_consecutive_failures
            and len(set(self._recent_codes[-self.max_consecutive_failures:])) == 1
        )
        if is_repeated:
            hint = (
                "⚠️ You have been executing the exact same code repeatedly without making progress. "
                "Stop and try a completely different approach.\n\n"
            )
            self.messages.append({"role": "user", "content": hint})

        try:
            output = self.language_model.generate(
                messages=self.trimmed_messages, max_tokens=self.max_output_tokens
            )
        except Exception as e:
            err = str(e)
            m = re.search(r"prompt contains at least (\d+) input tokens", err)
            if m and self.max_model_len is not None:
                actual = int(m.group(1))
                fallback_max_tokens = max(64, self.max_model_len - actual - 100)
                print(f"[context] 400 error: prompt={actual} tokens, retrying with max_tokens={fallback_max_tokens}")
                output = self.language_model.generate(
                    messages=self.trimmed_messages, max_tokens=fallback_max_tokens
                )
            else:
                raise

        code, fixed_content = self.extract_code_and_fix_content(output["content"])
        self._recent_codes.append(code.strip())
        self.messages.append({"role": "assistant", "content": fixed_content + "\n\n"})
        self.logger.show_message(
            role="agent", message=fixed_content, step_number=self.step_number
        )
        return [ExecutionIO(content=code)], output["cost"], None
