"""
Mem^p base agent.

Differences from ACE's StarAgent:
  - Two-layer memory: trajectory (concrete) + script (abstract) per entry
  - AveFact retrieval: keyword-level similarity averaging instead of full-text injection
  - Adjustment: failures trigger in-place script updates, not just ADD
  - Continuous update: memory is updated after every task (train AND test)
  - Model-agnostic: memory stored as plain JSON, readable by any model
"""

import os
from dataclasses import dataclass, field
from typing import Any

from appworld import AppWorld
from appworld.common.constants import DEFAULT_EXPERIMENT_NAME
from appworld.common.random import set_random_seed
from appworld.common.utils import FromDict, chunk_and_return
from appworld.evaluator import evaluate_task

from appworld_experiments.code.ace.cost_tracker import CostTracker
from appworld_experiments.code.ace.lite_llm_generator import LiteLLMGenerator
from appworld_experiments.code.ace.logger import Logger
from appworld_experiments.code.ace.memp.memory_store import MemoryEntry, MemoryStore
from appworld_experiments.code.ace.memp.retriever import AveFact, extract_keywords_simple


@dataclass
class ExecutionIO:
    content: str
    metadata: dict = field(default_factory=dict)


class MempAgent(FromDict):
    """
    Base class for Mem^p agents.

    Subclasses must implement:
      - extract_keywords(task_desc) -> list[str]
      - build_prompt(world)
      - next_execution_inputs_and_cost(last_outputs) -> (inputs, cost, reflection)
      - proceduralize(task_desc, trajectory, success) -> str
      - adjust_memory(failed_task_desc, trajectory, existing_script) -> str
    """

    def __init__(
        self,
        generator_model_config: dict,
        proceduralize_model_config: dict,
        adjust_model_config: dict,
        keyword_model_config: dict,
        memory_store_path: str,
        appworld_config: dict | None = None,
        logger_config: dict | None = None,
        max_steps: int = 40,
        max_cost_overall: float = 3000,
        max_cost_per_task: float = 10,
        log_lm_calls: bool = False,
        retrieval_top_k: int = 3,
    ):
        self.generator_model = LiteLLMGenerator(**generator_model_config)
        self.proceduralize_model = LiteLLMGenerator(**proceduralize_model_config)
        self.adjust_model = LiteLLMGenerator(**adjust_model_config)
        self.keyword_model = LiteLLMGenerator(**keyword_model_config)

        self.memory_store = MemoryStore(memory_store_path)
        self.retriever = AveFact(self.memory_store, top_k=retrieval_top_k)

        self.messages: list = []
        self.max_steps = max_steps
        self.step_number = 0
        self.appworld_config = appworld_config or {}
        self.random_seed = self.appworld_config.get("random_seed", None)
        self.cost_tracker = CostTracker(
            overall_limit=max_cost_overall, per_task_limit=max_cost_per_task
        )
        self.log_lm_calls = log_lm_calls
        logger_config = logger_config or {}
        logger_config["cost_tracker"] = self.cost_tracker
        self.logger = Logger(**logger_config)

        self.current_task_index = 0
        self.current_task_keywords: list = []
        self.retrieved_memories: list = []
        self.test_report: str | None = None

    def initialize(self, world: AppWorld):
        self.world = world
        if self.log_lm_calls:
            self.generator_model.log_calls_to(world=world)
        self.cost_tracker.reset(world.task_id)
        self.step_number = 0
        self.messages = []
        self.test_report = None
        self.logger.start_task(world)
        set_random_seed(self.random_seed)

    def extract_keywords(self, task_desc: str) -> list:
        raise NotImplementedError

    def build_prompt(self, world: AppWorld) -> None:
        raise NotImplementedError

    def next_execution_inputs_and_cost(
        self, last_execution_outputs: list
    ) -> tuple:
        raise NotImplementedError

    def proceduralize(self, task_desc: str, trajectory: list, success: bool) -> str:
        raise NotImplementedError

    def adjust_memory(
        self, failed_task_desc: str, trajectory: list, existing_script: str
    ) -> str:
        raise NotImplementedError

    def build_memory_context(self, retrieved: list) -> str:
        """Format retrieved memory entries for injection into the generator prompt."""
        if not retrieved:
            return "(No relevant past experience found in memory yet.)"
        parts = []
        for i, entry in enumerate(retrieved, 1):
            outcome = "SUCCESS" if entry.success else "FAILURE"
            header = f"[Memory {i} | {outcome}] Task type: {entry.task_desc[:100]}"
            parts.append(f"{header}\n{entry.script}")
        return "\n\n".join(parts)

    def solve_task(self, task_id: str, experiment_name: str | None = None):
        experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME
        self.cost_tracker.reset(task_id)
        task_desc = ""
        task_success = False

        with AppWorld(
            task_id=task_id, experiment_name=experiment_name, **self.appworld_config
        ) as world:
            self.initialize(world)
            task_desc = world.task.instruction

            # Concept 2 — AveFact: extract keywords, retrieve relevant memories
            self.current_task_keywords = self.extract_keywords(task_desc)
            self.retrieved_memories = self.retriever.retrieve(
                task_desc, self.current_task_keywords
            )
            print(
                f"[Mem^p] Retrieved {len(self.retrieved_memories)} memories "
                f"for task: {task_desc[:60]}..."
            )

            # Build generator prompt with retrieved memory context
            self.build_prompt(world)

            execution_outputs: list = []
            print("---Max steps---:", self.max_steps)
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
                    for output in execution_outputs:
                        if output.content.strip():
                            self.logger.show_message(
                                role="environment",
                                message=output.content,
                                step_number=self.step_number,
                            )

                self.cost_tracker.add(task_id, cost)
                self.log_cost()

                if world.task_completed() or self.cost_tracker.exceeded():
                    break

            # Evaluate inside the AppWorld context
            test_tracker, self.test_report = evaluate_task(task_id, experiment_name)
            task_success = len(test_tracker.failures) == 0

        # Concept 3 & 4 — update memory outside AppWorld (no env dependency)
        self.update_memory(task_desc, task_success)
        self.logger.complete_task()

    def update_memory(self, task_desc: str, success: bool) -> None:
        """
        Mem^p memory update cycle (Concepts 3 & 4):
          - SUCCESS (Validation): proceduralize trajectory → new script, store entry
          - FAILURE (Adjustment): update existing retrieved scripts in-place,
                                  then also store the failure experience
        """
        trajectory = list(self.messages)

        if success:
            script = self.proceduralize(task_desc, trajectory, success=True)
            entry = MemoryEntry.create(
                task_desc=task_desc,
                keywords=self.current_task_keywords,
                trajectory=trajectory,
                script=script,
                success=True,
            )
            self.memory_store.add(entry)
            print(f"[Mem^p] Stored SUCCESS memory (id={entry.id}): {task_desc[:60]}...")
        else:
            # Adjustment: refine existing relevant memories based on failure
            for existing in self.retrieved_memories:
                updated_script = self.adjust_memory(
                    failed_task_desc=task_desc,
                    trajectory=trajectory,
                    existing_script=existing.script,
                )
                self.memory_store.update_script(existing.id, updated_script)
                print(f"[Mem^p] Adjusted memory {existing.id} from failure signal")

            # Also store the failure experience as a new entry
            script = self.proceduralize(task_desc, trajectory, success=False)
            entry = MemoryEntry.create(
                task_desc=task_desc,
                keywords=self.current_task_keywords,
                trajectory=trajectory,
                script=script,
                success=False,
            )
            self.memory_store.add(entry)
            print(f"[Mem^p] Stored FAILURE memory (id={entry.id}): {task_desc[:60]}...")

    def solve_tasks(
        self,
        task_ids: list,
        experiment_name: str | None = None,
        num_processes: int = 1,
        process_index: int = 0,
    ):
        num_tasks = len(task_ids)
        num_processes = min(num_processes, num_tasks)
        task_ids = chunk_and_return(
            task_ids, num_chunks=num_processes, chunk_index=process_index
        )
        self.logger.initialize(
            experiment_name=experiment_name,
            num_tasks=num_tasks,
            num_processes=num_processes,
            process_index=process_index,
        )
        for task_index, task_id in enumerate(task_ids):
            self.current_task_index = task_index
            self.solve_task(task_id, experiment_name)

    def log_cost(self) -> None:
        self.cost_tracker.save(
            os.path.join(self.world.output_misc_directory, "cost.txt")
        )
