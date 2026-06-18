"""
SASM Teacher Agent (Adaptation Phase).

Purely SASM-based: no playbook, no reflector, no curator.
After each task, decomposes the trajectory into subtask segments and
extracts (z, d, e) triples into the SASM memory bank.
"""

import json
import os
from typing import Any

from appworld import AppWorld
from appworld.common.utils import read_file
from appworld.evaluator import evaluate_task
from appworld_experiments.code.ace.adaptation_agent import StarAgent, ExecutionIO
from appworld_experiments.code.ace.playbook import extract_json_from_text
from appworld_experiments.code.ace.sasm_memory import SASMMemoryBank
from appworld_experiments.code.ace.sasm_react_base import SASMReActMixin


@StarAgent.register("sasm_adaptation_react")
class SASMAdaptationReActAgent(SASMReActMixin, StarAgent):
    """
    Teacher agent that builds SASM memory from successful trajectories.
    Inherits the solve_task/solve_tasks loop from StarAgent.
    Uses SASMReActMixin for ReAct mechanics (no playbook).
    """

    def __init__(
        self,
        generator_model_config: dict,
        extractor_model_config: dict,
        generator_prompt_file_path: str,
        sasm_memory_file_path: str,
        sasm_decomposer_prompt_file_path: str,
        sasm_extractor_prompt_file_path: str,
        ignore_multiple_calls: bool = True,
        max_prompt_length: int | None = None,
        max_output_length: int = 400000,
        **kwargs: Any,
    ):
        # Pass only what StarAgent needs (no reflector/curator models)
        super().__init__(
            generator_model_config=generator_model_config,
            reflector_model_config=extractor_model_config,
            curator_model_config=extractor_model_config,
            **kwargs,
        )
        self.generator_prompt_template = read_file(
            generator_prompt_file_path.replace("/", os.sep)
        ).lstrip()
        self.memory_bank = SASMMemoryBank(sasm_memory_file_path)
        self.sasm_decomposer_prompt = read_file(
            sasm_decomposer_prompt_file_path.replace("/", os.sep)
        )
        self.sasm_extractor_prompt = read_file(
            sasm_extractor_prompt_file_path.replace("/", os.sep)
        )
        self.ignore_multiple_calls = ignore_multiple_calls
        self.max_prompt_length = max_prompt_length
        self.max_output_length = max_output_length

        # Disable playbook (StarAgent sets self.playbook = '')
        self.playbook = ""
        self._last_task_succeeded = False

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def initialize(self, world: AppWorld) -> None:
        super().initialize(world)
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

    # ------------------------------------------------------------------
    # ReAct step
    # ------------------------------------------------------------------

    def next_execution_inputs_and_cost(
        self,
        last_execution_outputs: list[ExecutionIO],
        world_gt_code: str = None,
        reasoning_text: str = "",
    ) -> tuple[list[ExecutionIO], float, str | None]:
        if last_execution_outputs:
            assert len(last_execution_outputs) == 1
            out_content = (
                "Output:\n```\n"
                + self._truncate_output(last_execution_outputs[0].content)
                + "```\n\n"
            )
            self.messages.append({"role": "user", "content": out_content})

        output = self.generator_model.generate(messages=self.trimmed_messages)
        code, fixed_content = self.extract_code_and_fix_content(output["content"])
        self.messages.append({"role": "assistant", "content": fixed_content + "\n\n"})
        self.logger.show_message(
            role="agent", message=fixed_content, step_number=self.step_number
        )
        return [ExecutionIO(content=code)], output["cost"], None

    # ------------------------------------------------------------------
    # SASM memory extraction (called after task completes)
    # ------------------------------------------------------------------

    def _build_trajectory_text(self) -> str:
        lines = []
        for i, msg in enumerate(self.trimmed_messages):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"[Step {i}] {role}:\n{content}\n")
        return "\n".join(lines)

    def _decompose_trajectory(self, task_instruction: str) -> list[dict]:
        trajectory_text = self._build_trajectory_text()
        prompt = (
            self.sasm_decomposer_prompt
            .replace("{{task_instruction}}", task_instruction)
            .replace("{{trajectory}}", trajectory_text)
        )
        response = self.curator_model.generate(
            messages=[{"role": "user", "content": prompt}]
        )
        parsed = extract_json_from_text(response.get("content", ""))
        if not parsed or "subtasks" not in parsed:
            print("[SASM] Decomposer returned invalid JSON; skipping memory extraction.")
            return []
        subtasks = parsed["subtasks"]
        print(f"[SASM] Decomposed into {len(subtasks)} subtasks.")
        return subtasks

    def _extract_experience(
        self, z: str, d: str, start_step: int, end_step: int, task_succeeded: bool
    ) -> str | None:
        msgs = self.trimmed_messages[start_step: end_step + 1]
        segment_text = "\n".join(
            f"{m.get('role','').upper()}:\n{m.get('content','')}" for m in msgs
        )
        prompt = (
            self.sasm_extractor_prompt
            .replace("{{z}}", z)
            .replace("{{d}}", d)
            .replace("{{trajectory_segment}}", segment_text)
            .replace("{{task_succeeded}}", "true" if task_succeeded else "false")
        )
        response = self.curator_model.generate(
            messages=[{"role": "user", "content": prompt}]
        )
        parsed = extract_json_from_text(response.get("content", ""))
        if not parsed or "e" not in parsed:
            print(f"[SASM] Extractor returned invalid JSON for z={z}; skipping.")
            return None
        return parsed["e"]

    def _build_sasm_memories(self, task_instruction: str, task_succeeded: bool) -> None:
        outcome = "SUCCESS" if task_succeeded else "FAILURE"
        print(f"[SASM] Building memory entries from {outcome} trajectory...")
        subtasks = self._decompose_trajectory(task_instruction)
        added = 0
        for subtask in subtasks:
            z = subtask.get("z", "")
            d = subtask.get("d", "")
            start_step = subtask.get("start_step", 0)
            end_step = subtask.get("end_step", 0)
            if not z or not d:
                continue
            e = self._extract_experience(z, d, start_step, end_step, task_succeeded)
            if e:
                self.memory_bank.add(z=z, d=d, e=e)
                added += 1
                print(f"[SASM] Stored ({outcome}): z={z}, d={d[:60]}...")
        print(f"[SASM] Added {added} entries. Total: {self.memory_bank.stats()}")

    # ------------------------------------------------------------------
    # Override solve_task_wo_gt: record whether task succeeded
    # ------------------------------------------------------------------

    def solve_task_wo_gt(self, task_id: str, experiment_name: str | None = None) -> None:
        self._last_task_succeeded = False
        self.star_guide_idx = None
        self.initial_code_idx = None
        self.previous_code_idx = None
        self.previous_error_idx = None
        self.test_report = None

        with AppWorld(
            task_id=task_id, experiment_name=experiment_name, **self.appworld_config
        ) as world:
            execution_outputs: list[ExecutionIO] = []
            self.initialize(world)
            print("---Max steps---: ", self.max_steps)
            for _ in range(self.max_steps):
                self.step_number += 1
                execution_inputs, cost, _ = self.next_execution_inputs_and_cost(
                    execution_outputs
                )
                if execution_inputs:
                    execution_outputs = [
                        ExecutionIO(
                            content=world.execute(ei.content),
                            metadata=ei.metadata,
                        )
                        for ei in execution_inputs
                    ]
                    for out in execution_outputs:
                        if out.content.strip():
                            self.logger.show_message(
                                role="environment",
                                message=out.content,
                                step_number=self.step_number,
                            )
                self.cost_tracker.add(task_id, cost)
                self.log_cost()
                if world.task_completed() or self.cost_tracker.exceeded():
                    test_tracker, self.test_report = evaluate_task(
                        task_id, experiment_name
                    )
                    self._last_task_succeeded = len(test_tracker.failures) == 0
                    self.curator_call()
                    break

        if (self.current_task_index + 1) % 30 == 0:
            self.save_playbook_snapshot()

        self.logger.complete_task()

    # ------------------------------------------------------------------
    # Override curator_call: SASM only, no playbook
    # Only builds memory when the teacher solved the task successfully.
    # ------------------------------------------------------------------

    def curator_call(self) -> None:
        task_instruction = getattr(
            getattr(self, "world", None), "task", None
        )
        task_instruction = (
            getattr(task_instruction, "instruction", "") if task_instruction else ""
        )
        if task_instruction:
            try:
                self._build_sasm_memories(task_instruction, self._last_task_succeeded)
            except Exception as exc:
                print(f"[SASM] Memory extraction failed: {exc}")
