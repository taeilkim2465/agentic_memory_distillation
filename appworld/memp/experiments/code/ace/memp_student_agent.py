"""
Mem^p student agent — retrieval-only, no memory updates.

The student uses the memory built by the teacher:
  - Loads the same memory store (read-only)
  - Applies AveFact retrieval before each task
  - Injects retrieved memory context into the generator prompt
  - Does NOT call proceduralize / adjust / update_memory

This enables Concept 5 of the Mem^p paper:
  a stronger teacher model builds memory that a weaker student can exploit.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from appworld import AppWorld
from appworld.common.constants import DEFAULT_EXPERIMENT_NAME
from appworld.common.random import set_random_seed
from appworld.common.utils import FromDict, chunk_and_return

from appworld_experiments.code.ace.cost_tracker import CostTracker
from appworld_experiments.code.ace.lite_llm_generator import LiteLLMGenerator
from appworld_experiments.code.ace.logger import Logger
from appworld_experiments.code.ace.memp.memory_store import MemoryStore
from appworld_experiments.code.ace.memp.retriever import AveFact, extract_keywords_simple


@dataclass
class ExecutionIO:
    content: str
    metadata: dict = field(default_factory=dict)


class MempStudentAgent(FromDict):
    """
    Read-only Mem^p agent. Retrieves from teacher's memory store, never writes.

    Required model configs:
      - generator_model_config : solves the task
      - keyword_model_config   : extracts keywords for AveFact retrieval
    """

    def __init__(
        self,
        generator_model_config: dict,
        keyword_model_config: dict,
        memory_store_path: str,
        appworld_config: dict | None = None,
        logger_config: dict | None = None,
        max_steps: int = 40,
        max_cost_overall: float = 3000,
        max_cost_per_task: float = 10,
        log_lm_calls: bool = False,
        retrieval_top_k: int = 3,
        retrieval_embedding_model: str | None = None,
        retrieval_embedding_api_key: str | None = None,
        retrieval_embedding_base_url: str | None = None,
    ):
        self.generator_model = LiteLLMGenerator(**generator_model_config)
        self.keyword_model = LiteLLMGenerator(**keyword_model_config)

        self.memory_store = MemoryStore(memory_store_path)
        self.retriever = AveFact(
            self.memory_store,
            top_k=retrieval_top_k,
            embedding_model=retrieval_embedding_model,
            embedding_api_key=retrieval_embedding_api_key,
            embedding_base_url=retrieval_embedding_base_url,
        )

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

    def initialize(self, world: AppWorld):
        self.world = world
        if self.log_lm_calls:
            self.generator_model.log_calls_to(world=world)
        self.cost_tracker.reset(world.task_id)
        self.step_number = 0
        self.messages = []
        self.logger.start_task(world)
        set_random_seed(self.random_seed)

    def extract_keywords(self, task_desc: str) -> list:
        raise NotImplementedError

    def build_prompt(self, world: AppWorld) -> None:
        raise NotImplementedError

    def next_execution_inputs_and_cost(self, last_execution_outputs: list) -> tuple:
        raise NotImplementedError

    def build_memory_context(self, retrieved: list) -> str:
        if not retrieved:
            return "(No relevant past experience found in memory.)"
        parts = []
        for i, entry in enumerate(retrieved, 1):
            outcome = "SUCCESS" if entry.success else "FAILURE"
            header = f"[Memory {i} | {outcome}] Task type: {entry.task_desc[:100]}"
            parts.append(f"{header}\n{entry.script}")
        return "\n\n".join(parts)

    def solve_task(self, task_id: str, experiment_name: str | None = None):
        experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME
        self.cost_tracker.reset(task_id)

        with AppWorld(
            task_id=task_id, experiment_name=experiment_name, **self.appworld_config
        ) as world:
            self.initialize(world)
            task_desc = world.task.instruction

            # AveFact: extract keywords and retrieve relevant teacher memories
            self.current_task_keywords = self.extract_keywords(task_desc)
            self.retrieved_memories = self.retriever.retrieve(
                task_desc, self.current_task_keywords
            )
            print(
                f"[Mem^p Student] Retrieved {len(self.retrieved_memories)} memories "
                f"from teacher for: {task_desc[:60]}..."
            )

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

        # No memory update — student is read-only
        self.logger.complete_task()

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
