import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from bfcl_eval.constants.category_mapping import get_version_prefix
from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
    MAXIMUM_STEP_LIMIT,
)
from bfcl_eval.constants.enums import ModelStyle, ReturnFormat
from bfcl_eval.constants.eval_config import RESULT_PATH
from bfcl_eval.constants.executable_backend_config import (
    OMIT_STATE_INFO_CLASSES,
    STATELESS_CLASSES,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.model_handler.utils import add_memory_instruction_system_prompt
from bfcl_eval.utils import *
from overrides import final

if TYPE_CHECKING:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_api_metaclass import (
        MemoryAPI,
    )


_THINK_SYSTEM = (
    "You generate concise reasoning for function-calling agents."
)

_THINK_USER_TEMPLATE = """\
You are an expert agent solving a multi-turn function-calling task.
Before making a function call, reason about what to do next.

{prev_section}\
Current turn: {current_instruction}

Available functions: {tool_names}

In 2-3 sentences:
1. Which function should be called and what it provides for this task
2. What prerequisite information is already available from prior calls
3. Key constraints on the arguments to keep in mind

Reply with only the reasoning, no bullets, no markdown.
"""


class BaseHandler:
    model_name: str
    is_fc_model: bool
    registry_name: str
    temperature: float
    registry_dir_name: str
    model_name_underline_replaced: str
    model_style: ModelStyle

    def __init__(
        self, model_name, temperature, registry_name, is_fc_model, **kwargs
    ) -> None:
        """
        Args:
            model_name: The name of the model as used in the vendor API or on Hugging Face.
            temperature: The temperature of the model.
            registry_name: The name of the model as used internally in BFCL, used for result directory naming.
            is_fc_model: Whether the model is a function calling model.
            **kwargs: Additional attributes passed via kwargs.
        """
        self.model_name = model_name
        self.is_fc_model = is_fc_model
        self.registry_name = registry_name

        # Replace the dash and dot with underscore for valid variable name
        self.model_name_underline_replaced = (
            model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        )
        # The directory name for the model
        # Replace the slash with underscore to avoid creating subdirectories
        self.registry_dir_name = registry_name.replace("/", "_")
        self.temperature = temperature

        # Optional: memory directory for tau2-style teacher/student memory injection.
        # Set via --memory-dir CLI flag; None means memory is disabled.
        _memory_dir = kwargs.pop("memory_dir", None)
        self.memory_dir: Optional[Path] = Path(_memory_dir) if _memory_dir else None

        # LLM used for subtask decomposition during memory retrieval (small/fast model).
        self.decomposer_llm: str = kwargs.pop("decomposer_llm", "gpt-4o-mini")
        self.subtask_source: str = kwargs.pop("subtask_source", "turns")

        # Comma-separated set of active memory types: "workflow", "subtask", "function".
        _memory_types = kwargs.pop("memory_types", "workflow,subtask,function")
        self.memory_types: set[str] = {t.strip() for t in _memory_types.split(",") if t.strip()}

        # "static": inject workflow+subtask memory once before Turn 1 (default).
        # "dynamic": retrieve subtask memory per turn using that turn's user instruction.
        self.memory_retrieval: str = kwargs.pop("memory_retrieval", "static")

        # LLM used to generate "think" annotations before each tool call (teacher-only).
        # When set, the model generates WHY each function is called; stored in result JSON.
        self.think_model: Optional[str] = kwargs.pop("think_model", None)

        # Set any additional attributes passed via kwargs
        for _key, _value in kwargs.items():
            setattr(self, _key, _value)

    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        # This method is used to retrive model response for each model.

        # FC model
        # TODO: Let all models have the is_fc_model attribute and remove the "FC" check
        if "FC" in self.registry_name or self.is_fc_model:
            if contain_multi_turn_interaction(test_entry["id"]):
                return self.inference_multi_turn_FC(
                    test_entry, include_input_log, exclude_state_log
                )
            else:
                return self.inference_single_turn_FC(test_entry, include_input_log)
        # Prompting model
        else:
            if contain_multi_turn_interaction(test_entry["id"]):
                return self.inference_multi_turn_prompting(
                    test_entry, include_input_log, exclude_state_log
                )
            else:
                return self.inference_single_turn_prompting(test_entry, include_input_log)

    @final
    def inference_multi_turn_FC(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ) -> tuple[list[list], dict]:
        initial_config: dict = test_entry.get("initial_config", {})
        involved_classes: list = test_entry["involved_classes"]
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        # This is only for the miss function category
        # A mapping from turn index to function to holdout
        holdout_function: dict[int, list] = test_entry.get("missed_function", {})

        total_input_token_count: list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_latency: list[list[float]] = []
        all_model_response: list[list] = (
            []
        )  # The model response that will be used for later evaluation
        all_inference_log: list[list[dict]] = (
            []
        )  # The debugging log for human to understand
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed.
        has_decode_error = False

        all_reasoning_content: list[list] = []

        # Execute no function call, but just to get a reference to all the instances to get the initial state for logging purpose
        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            test_entry_id,
            long_context=("long_context" in test_category or "composite" in test_category),
            is_evaL_run=False,
        )

        if is_memory(test_category):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."

            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            test_entry["question"] = add_memory_instruction_system_prompt(
                test_entry["question"],
                test_category,
                test_entry["scenario"],
                memory_instance,
            )

        # Workflow memory injected statically before Turn 1.
        # dynamic/system_replace: subtask excluded here, retrieved per turn instead.
        _per_turn = self.memory_retrieval in ("dynamic", "system_replace")
        static_types = {"workflow"} if _per_turn else {"workflow", "subtask"}
        test_entry = self._apply_static_memory(test_entry, override_types=static_types)

        if not exclude_state_log:
            state_log = []
            for class_name, class_instance in involved_instances.items():
                if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_INFO_CLASSES:
                    continue
                # Avoid modification in future turns
                class_instance = deepcopy(class_instance)
                state_log.append(
                    {
                        "role": "state_info",
                        "class_name": class_name,
                        "content": {
                            key: value
                            for key, value in vars(class_instance).items()
                            if not key.startswith("_")
                        },
                    }
                )
            if len(state_log) > 0:
                all_inference_log.append(state_log)

        _C_RESET  = "\033[0m"
        _C_BOLD   = "\033[1m"
        _C_BLUE   = "\033[94m"
        _C_CYAN   = "\033[96m"
        _C_GREEN  = "\033[92m"
        _C_YELLOW = "\033[93m"
        _C_RED    = "\033[91m"
        _C_GRAY   = "\033[90m"

        total_turns = len(test_entry["question"])
        short_id = test_entry_id.replace("multi_turn_", "")

        inference_data: dict = {}
        inference_data = self._pre_query_processing_FC(inference_data, test_entry)
        inference_data = self._compile_tools(inference_data, test_entry)

        # base_system_content is captured after add_first_turn_message_FC (Turn 0),
        # because the system message dict is created inside that call (not before).
        base_system_content: Optional[str] = None

        all_multi_turn_messages: list[list[dict]] = test_entry["question"]
        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_message: list[dict]

            if str(turn_idx) in holdout_function:
                test_entry["function"].extend(holdout_function[str(turn_idx)])
                # Since we have added new functions, we need to recompile the tools
                inference_data = self._compile_tools(inference_data, test_entry)
                assert (
                    len(current_turn_message) == 0
                ), "Holdout turn should not have user message."
                # TODO: Move this to before pre_query_processing_FC.
                # Shouldn't be happening in the inference loop.
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                    }
                ]

            turn_instruction = next(
                (m["content"] for m in current_turn_message if m.get("role") == "user"), ""
            )

            if self.memory_retrieval == "dynamic":
                current_turn_message = self._apply_dynamic_memory_for_turn(
                    current_turn_message, test_entry
                )

            if turn_idx == 0:
                inference_data = self.add_first_turn_message_FC(
                    inference_data, current_turn_message
                )
                # system message dict is created inside add_first_turn_message_FC → capture now
                if self.memory_retrieval == "system_replace":
                    base_system_content = self._get_system_content(inference_data)
            else:
                inference_data = self._add_next_turn_user_message_FC(
                    inference_data, current_turn_message
                )

            if self.memory_retrieval == "system_replace":
                inference_data = self._replace_system_prompt_subtask(
                    inference_data, base_system_content, turn_instruction, test_entry
                )

            # SASM: predict subtask → retrieve experience → inject (turn-level)
            _sasm_injected = 0
            if getattr(self, "sasm_pipeline", None) is not None:
                _sasm_injected = self.sasm_pipeline.pre_turn_inject(
                    turn_idx,
                    current_turn_message,
                    inference_data,
                    getattr(self, "sasm_test_entry", test_entry),
                )

            current_turn_response = []
            current_turn_inference_log: list[dict] = {
                "begin_of_turn_query": current_turn_message
            }
            current_turn_input_token_count: list[float] = []
            current_turn_output_token_count: list[float] = []
            current_turn_latency: list[float] = []
            current_turn_reasoning_content = []

            user_content = current_turn_message[0]["content"] if current_turn_message else ""
            print(f"\n{_C_BOLD}{_C_BLUE}{'─' * 80}{_C_RESET}")
            print(f"{_C_BOLD}{_C_BLUE}[ ID: {short_id} │ Turn {turn_idx + 1}/{total_turns} ]{_C_RESET}")
            print(f"{_C_GRAY}  User: {user_content[:120]}{'...' if len(user_content) > 120 else ''}{_C_RESET}")

            # Generate think once per turn (teacher-only).
            # The teacher reasons about what functions this turn will need and why,
            # but this reasoning is NOT injected into the model context.
            if self.think_model:
                turn_think_text = self._generate_pre_call_think(
                    turn_idx,
                    all_multi_turn_messages,
                    all_model_response,
                    current_turn_message,
                    test_entry,
                )
                if turn_think_text:
                    current_turn_inference_log["turn_think"] = turn_think_text

            count = 0
            while True:
                print(f"{_C_CYAN}  ┌─ Step {count} {'─' * 40}{_C_RESET}")
                current_step_inference_log: list[dict] = []
                # Add to the current_turn_inference_log at beginning of each step so that we don't need to bother dealing with the break statements
                current_turn_inference_log[f"step_{count}"] = current_step_inference_log

                api_response, query_latency = self._query_FC(inference_data)

                # This part of logging is disabled by default because it is too verbose and will make the result file extremely large
                # It is only useful to see if the inference pipeline is working as expected (eg, does it convert all the inputs correctly)
                if include_input_log:
                    current_step_inference_log.append(
                        {
                            "role": "inference_input",
                            "content": inference_data.get("inference_input_log", ""),
                        }
                    )

                # Try parsing the model response
                model_response_data = self._parse_query_response_FC(api_response)
                model_responses = model_response_data["model_responses"]
                in_tok  = model_response_data.get("input_token", 0)
                out_tok = model_response_data.get("output_token", 0)
                reasoning = model_response_data.get("reasoning_content", "")
                print(f"{_C_GREEN}  │  Tool call : {model_responses}{_C_RESET}")
                if not model_responses and reasoning:
                    preview = reasoning[:200].replace("\n", " ")
                    print(f"{_C_GRAY}  │  Think     : {preview}{'...' if len(reasoning) > 200 else ''}{_C_RESET}")
                print(f"{_C_GRAY}  │  Perf      : {query_latency:.2f}s │ in={in_tok} out={out_tok} tok{_C_RESET}")

                # Add the assistant message to the chat history
                inference_data = self._add_assistant_message_FC(
                    inference_data, model_response_data
                )

                # Process the metadata
                current_turn_input_token_count.append(model_response_data["input_token"])
                current_turn_output_token_count.append(model_response_data["output_token"])
                current_turn_latency.append(query_latency)

                current_turn_response.append(model_responses)

                reasoning_content = model_response_data.get("reasoning_content", "")
                current_turn_reasoning_content.append(reasoning_content)

                log_entry = {
                    "role": "assistant",
                    "content": model_responses,
                }
                if reasoning_content:
                    log_entry["reasoning_content"] = reasoning_content

                current_step_inference_log.append(log_entry)

                # Try decoding the model response
                try:
                    decoded_model_responses = self.decode_execute(
                        model_responses, has_tool_call_tag=False
                    )
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_model_responses,
                        }
                    )

                    if is_empty_execute_response(decoded_model_responses):
                        print(f"{_C_GRAY}  └─ (no tool call, turn complete){_C_RESET}")
                        current_step_inference_log.append(
                            {
                                "role": "handler_log",
                                "content": f"Empty response from the model. Proceed to next turn.",
                                "model_response_decoded": decoded_model_responses,
                            }
                        )
                        break

                except Exception as e:
                    has_decode_error = True
                    print(f"{_C_RED}  └─ Decode error: {e}{_C_RESET}")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Error decoding the model response. Proceed to next turn.",
                            "error": str(e),
                        }
                    )
                    break

                # Obtain the execution results
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    self.model_name_underline_replaced,
                    test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=False,
                )

                # Dynamic function memory: append hints to error results
                execution_results = self._augment_execution_results_with_hints(
                    execution_results, decoded_model_responses, current_turn_message, test_entry
                )

                # Add the execution results to the chat history for the next turn
                inference_data = self._add_execution_results_FC(
                    inference_data, execution_results, model_response_data
                )

                for execution_result in execution_results:
                    print(f"{_C_YELLOW}  │  Result    : {execution_result}{_C_RESET}")
                    current_step_inference_log.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                        }
                    )

                count += 1
                # Force quit after too many steps
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    print(f"{_C_RED}  └─ Force quit: exceeded {MAXIMUM_STEP_LIMIT} steps{_C_RESET}")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                        }
                    )

                    break

            # SASM: remove injected message so it doesn't leak into next turn history
            if _sasm_injected > 0:
                inference_data["message"] = inference_data["message"][:-_sasm_injected]

            # Add to the total list
            all_model_response.append(current_turn_response)
            all_inference_log.append(current_turn_inference_log)
            all_reasoning_content.append(current_turn_reasoning_content)
            total_input_token_count.append(current_turn_input_token_count)
            total_output_token_count.append(current_turn_output_token_count)
            total_latency.append(current_turn_latency)

            if not exclude_state_log:
                state_log = []
                for class_name, class_instance in involved_instances.items():
                    if (
                        class_name in STATELESS_CLASSES
                        or class_name in OMIT_STATE_INFO_CLASSES
                    ):
                        continue
                    # Avoid modification in future turns
                    class_instance = deepcopy(class_instance)
                    state_log.append(
                        {
                            "role": "state_info",
                            "class_name": class_name,
                            "content": {
                                key: value
                                for key, value in vars(class_instance).items()
                                if not key.startswith("_")
                            },
                        }
                    )
                if len(state_log) > 0:
                    all_inference_log.append(state_log)

            if force_quit:
                break

        # Special handling for the memory category
        # Need to flush the memory to local file at the end of the conversation
        if is_memory_prereq(test_entry_id):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."
            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            memory_instance._flush_memory_to_local_file()

        if force_quit:
            generation_status = "force_quit"
        elif has_decode_error:
            generation_status = "decode_error"
        else:
            generation_status = "ok"

        metadata = {
            "input_token_count": total_input_token_count,
            "output_token_count": total_output_token_count,
            "latency": total_latency,
            "inference_log": all_inference_log,
            "generation_status": generation_status,
        }

        if not all(
            all(content == "" for content in single_turn_reasoning_content)
            for single_turn_reasoning_content in all_reasoning_content
        ):
            metadata["reasoning_content"] = all_reasoning_content

        return all_model_response, metadata

    @final
    def inference_multi_turn_prompting(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ) -> tuple[list[list], dict]:
        initial_config: dict = test_entry.get("initial_config", {})
        involved_classes: list = test_entry["involved_classes"]
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        # This is only for the miss function category
        # A mapping from turn index to function to holdout
        holdout_function: dict[int, list] = test_entry.get("missed_function", {})

        total_input_token_count: list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_latency: list[list[float]] = []
        # The model response that will be used for later evaluation
        all_model_response: list[list] = []
        # Only for reasoning models, reasoning content will be stored as part of metadata and in inference log
        all_reasoning_content: list[list] = []
        # The debugging log for human to understand
        all_inference_log: list[list[dict]] = []
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed.
        has_decode_error = False

        # Execute no function call, but just to get a reference to all the instances to get the initial state for logging purpose
        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            test_entry_id,
            long_context=("long_context" in test_category or "composite" in test_category),
            is_evaL_run=False,
        )

        if is_memory(test_category):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."

            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            test_entry["question"] = add_memory_instruction_system_prompt(
                test_entry["question"],
                test_category,
                test_entry["scenario"],
                memory_instance,
            )

        if not exclude_state_log:
            state_log = []
            for class_name, class_instance in involved_instances.items():
                if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_INFO_CLASSES:
                    continue
                # Avoid modification in future turns
                class_instance = deepcopy(class_instance)
                state_log.append(
                    {
                        "role": "state_info",
                        "class_name": class_name,
                        "content": {
                            key: value
                            for key, value in vars(class_instance).items()
                            if not key.startswith("_")
                        },
                    }
                )
            if len(state_log) > 0:
                all_inference_log.append(state_log)

        _per_turn = self.memory_retrieval in ("dynamic", "system_replace")
        static_types = {"workflow"} if _per_turn else {"workflow", "subtask"}
        test_entry = self._apply_static_memory(test_entry, override_types=static_types)

        inference_data: dict = self._pre_query_processing_prompting(test_entry)

        _C_RESET  = "\033[0m"
        _C_BOLD   = "\033[1m"
        _C_BLUE   = "\033[94m"
        _C_CYAN   = "\033[96m"
        _C_GREEN  = "\033[92m"
        _C_YELLOW = "\033[93m"
        _C_RED    = "\033[91m"
        _C_GRAY   = "\033[90m"

        total_turns = len(test_entry["question"])
        short_id = test_entry_id.replace("multi_turn_", "")

        base_system_content: Optional[str] = None

        all_multi_turn_messages: list[list[dict]] = test_entry["question"]
        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_message: list[dict]

            if str(turn_idx) in holdout_function:
                assert (
                    len(current_turn_message) == 0
                ), "Holdout turn should not have user message."
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                            functions=holdout_function[str(turn_idx)]
                        ),
                    }
                ]

            turn_instruction = next(
                (m["content"] for m in current_turn_message if m.get("role") == "user"), ""
            )

            if self.memory_retrieval == "dynamic":
                current_turn_message = self._apply_dynamic_memory_for_turn(
                    current_turn_message, test_entry
                )

            if turn_idx == 0:
                inference_data = self.add_first_turn_message_prompting(
                    inference_data, current_turn_message
                )
                if self.memory_retrieval == "system_replace":
                    base_system_content = self._get_system_content(inference_data)
            else:
                inference_data = self._add_next_turn_user_message_prompting(
                    inference_data, current_turn_message
                )

            if self.memory_retrieval == "system_replace":
                inference_data = self._replace_system_prompt_subtask(
                    inference_data, base_system_content, turn_instruction, test_entry
                )

            # SASM: predict subtask → retrieve experience → inject (turn-level)
            _sasm_injected = 0
            if getattr(self, "sasm_pipeline", None) is not None:
                _sasm_injected = self.sasm_pipeline.pre_turn_inject(
                    turn_idx,
                    current_turn_message,
                    inference_data,
                    getattr(self, "sasm_test_entry", test_entry),
                )

            current_turn_response = []
            current_turn_reasoning_content = []
            current_turn_inference_log: list[dict] = {
                "begin_of_turn_query": current_turn_message
            }
            current_turn_input_token_count: list[float] = []
            current_turn_output_token_count: list[float] = []
            current_turn_latency: list[float] = []

            user_content = current_turn_message[0]["content"] if current_turn_message else ""
            print(f"\n{_C_BOLD}{_C_BLUE}{'─' * 80}{_C_RESET}")
            print(f"{_C_BOLD}{_C_BLUE}[ ID: {short_id} │ Turn {turn_idx + 1}/{total_turns} ]{_C_RESET}")
            print(f"{_C_GRAY}  User: {user_content[:120]}{'...' if len(user_content) > 120 else ''}{_C_RESET}")

            count = 0
            while True:
                print(f"{_C_CYAN}  ┌─ Step {count} {'─' * 40}{_C_RESET}")
                current_step_inference_log: list[dict] = []
                # Add to the current_turn_inference_log at beginning of each step so that we don't need to bother dealing with the break statements
                current_turn_inference_log[f"step_{count}"] = current_step_inference_log

                api_response, query_latency = self._query_prompting(inference_data)

                # This part of logging is disabled by default because it is too verbose and will make the result file extremely large
                # It is only useful to see if the inference pipeline is working as expected (eg, does it convert all the inputs correctly)
                if include_input_log:
                    current_step_inference_log.append(
                        {
                            "role": "inference_input",
                            "content": inference_data.get("inference_input_log", ""),
                        }
                    )

                # Try parsing the model response
                model_response_data = self._parse_query_response_prompting(api_response)
                model_responses = model_response_data["model_responses"]

                # Add the assistant message to the chat history
                inference_data = self._add_assistant_message_prompting(
                    inference_data, model_response_data
                )

                # Process the metadata
                current_turn_input_token_count.append(model_response_data["input_token"])
                current_turn_output_token_count.append(model_response_data["output_token"])
                current_turn_latency.append(query_latency)

                current_turn_response.append(model_responses)
                reasoning_content = model_response_data.get("reasoning_content", "")
                current_turn_reasoning_content.append(reasoning_content)

                log_entry = {
                    "role": "assistant",
                    "content": model_responses,
                }
                if reasoning_content:
                    log_entry["reasoning_content"] = reasoning_content

                current_step_inference_log.append(log_entry)

                # Try decoding the model response
                try:
                    decoded_model_responses = self.decode_execute(
                        model_responses, has_tool_call_tag=False
                    )
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_model_responses,
                        }
                    )

                    model_response_data["model_responses_decoded"] = decoded_model_responses
                    if is_empty_execute_response(decoded_model_responses):
                        print(f"{_C_GRAY}  └─ (no tool call, turn complete){_C_RESET}")
                        current_step_inference_log.append(
                            {
                                "role": "handler_log",
                                "content": f"Empty response from the model. Proceed to next turn.",
                                "model_response_decoded": decoded_model_responses,
                            }
                        )
                        break

                except Exception as e:
                    has_decode_error = True
                    print(f"{_C_RED}  └─ Decode error: {e}{_C_RESET}")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Error decoding the model response. Proceed to next turn.",
                            "error": str(e),
                        }
                    )
                    break

                # Obtain the execution results
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    self.model_name_underline_replaced,
                    test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=False,
                )

                # Add the execution results to the chat history for the next turn
                inference_data = self._add_execution_results_prompting(
                    inference_data, execution_results, model_response_data
                )

                for execution_result in execution_results:
                    print(f"{_C_YELLOW}  │  Result    : {execution_result}{_C_RESET}")
                    current_step_inference_log.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                        }
                    )

                count += 1
                # Force quit after too many steps
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    print(f"{_C_RED}  └─ Force quit: exceeded {MAXIMUM_STEP_LIMIT} steps{_C_RESET}")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                        }
                    )
                    break

            # SASM: remove injected message so it doesn't leak into next turn history
            if _sasm_injected > 0:
                inference_data["message"] = inference_data["message"][:-_sasm_injected]

            # Add to the total list
            all_model_response.append(current_turn_response)
            all_reasoning_content.append(current_turn_reasoning_content)
            all_inference_log.append(current_turn_inference_log)
            total_input_token_count.append(current_turn_input_token_count)
            total_output_token_count.append(current_turn_output_token_count)
            total_latency.append(current_turn_latency)

            if not exclude_state_log:
                state_log = []
                for class_name, class_instance in involved_instances.items():
                    if (
                        class_name in STATELESS_CLASSES
                        or class_name in OMIT_STATE_INFO_CLASSES
                    ):
                        continue
                    # Avoid modification in future turns
                    class_instance = deepcopy(class_instance)
                    state_log.append(
                        {
                            "role": "state_info",
                            "class_name": class_name,
                            "content": {
                                key: value
                                for key, value in vars(class_instance).items()
                                if not key.startswith("_")
                            },
                        }
                    )
                if len(state_log) > 0:
                    all_inference_log.append(state_log)

            if force_quit:
                break

        # Special handling for the memory category
        # Need to flush the memory to local file at the end of the conversation
        if is_memory_prereq(test_entry_id):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."
            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            memory_instance._flush_memory_to_local_file()

        if force_quit:
            generation_status = "force_quit"
        elif has_decode_error:
            generation_status = "decode_error"
        else:
            generation_status = "ok"

        metadata = {
            "input_token_count": total_input_token_count,
            "output_token_count": total_output_token_count,
            "latency": total_latency,
            "inference_log": all_inference_log,
            "generation_status": generation_status,
        }
        # We only include reasoning content if it exists and is not empty
        if not all(
            all(content == "" for content in single_turn_reasoning_content)
            for single_turn_reasoning_content in all_reasoning_content
        ):
            metadata["reasoning_content"] = all_reasoning_content

        return all_model_response, metadata

    @final
    def inference_single_turn_FC(
        self, test_entry: dict, include_input_log: bool
    ) -> tuple[any, dict]:
        inference_data: dict = {}
        inference_data = self._pre_query_processing_FC(inference_data, test_entry)
        inference_data = self._compile_tools(inference_data, test_entry)
        inference_data = self.add_first_turn_message_FC(
            inference_data, test_entry["question"][0]
        )

        api_response, query_latency = self._query_FC(inference_data)

        # Try parsing the model response
        model_response_data = self._parse_query_response_FC(api_response)

        # Process the metadata
        metadata = {}
        if include_input_log:
            metadata["inference_log"] = [
                {
                    "role": "inference_input",
                    "content": inference_data.get("inference_input_log", ""),
                }
            ]
        metadata["input_token_count"] = model_response_data["input_token"]
        metadata["output_token_count"] = model_response_data["output_token"]
        metadata["latency"] = query_latency

        if (
            "reasoning_content" in model_response_data
            and model_response_data["reasoning_content"] != ""
        ):
            metadata["reasoning_content"] = model_response_data["reasoning_content"]

        return model_response_data["model_responses"], metadata

    @final
    def inference_single_turn_prompting(
        self, test_entry: dict, include_input_log: bool
    ) -> tuple[any, dict]:
        inference_data: dict = self._pre_query_processing_prompting(test_entry)
        inference_data = self.add_first_turn_message_prompting(
            inference_data, test_entry["question"][0]
        )

        api_response, query_latency = self._query_prompting(inference_data)

        # Try parsing the model response
        model_response_data = self._parse_query_response_prompting(api_response)

        # Process the metadata
        metadata = {}
        if include_input_log:
            metadata["inference_log"] = [
                {
                    "role": "inference_input",
                    "content": inference_data.get("inference_input_log", ""),
                }
            ]
        metadata["input_token_count"] = model_response_data["input_token"]
        metadata["output_token_count"] = model_response_data["output_token"]
        metadata["latency"] = query_latency

        if (
            "reasoning_content" in model_response_data
            and model_response_data["reasoning_content"] != ""
        ):
            metadata["reasoning_content"] = model_response_data["reasoning_content"]

        return model_response_data["model_responses"], metadata

    def decode_ast(self, result, language: ReturnFormat, has_tool_call_tag: bool):
        """
        This method takes raw model output (from `_parse_query_response_xxx`) and convert it to standard AST checker input.
        """
        raise NotImplementedError

    def decode_execute(self, result, has_tool_call_tag: bool):
        """
        This method takes raw model output (from `_parse_query_response_xxx`) and convert it to standard execute checker input.
        """
        raise NotImplementedError

    @final
    def write(self, result, result_dir, update_mode=False):
        # Use the internal registry name to decide the result directory to avoid
        # collisions between different variants that share the same API model name.
        model_result_dir = result_dir / self.registry_dir_name

        if isinstance(result, dict):
            result = [result]

        # Collect and format each entry for JSON compatibility
        entries_to_write = [make_json_serializable(entry) for entry in result]

        # Group entries by their `test_category` for efficient file handling
        file_entries = {}
        for entry in entries_to_write:
            test_category = extract_test_category_from_id(entry["id"])
            # Determine the high-level grouping folder (non_live, live, etc.)
            group_dir_name = get_directory_structure_by_id(entry["id"])
            group_dir_path = model_result_dir / group_dir_name
            group_dir_path.mkdir(parents=True, exist_ok=True)

            file_path = group_dir_path / f"{get_version_prefix(test_category)}_{test_category}_result.json"
            file_entries.setdefault(file_path, []).append(entry)

        for file_path, entries in file_entries.items():
            if update_mode:
                # Load existing entries from the file
                existing_entries = {}
                if file_path.exists():
                    existing_entries = {
                        entry["id"]: entry for entry in load_file(file_path)
                    }

                # Update existing entries with new data
                for entry in entries:
                    existing_entries[entry["id"]] = entry

                # Sort entries by `id` and write them back to ensure order consistency
                sorted_entries = sorted(existing_entries.values(), key=sort_key)
                with open(file_path, "w") as f:
                    for entry in sorted_entries:
                        content = json.dumps(entry) + "\n"
                        f.write(content)
                        f.flush()

            else:
                # Normal mode: Append to the end of the file
                # Note: We will sort all the entries at the end of the generation pipeline to ensure the order is consistent
                entries.sort(key=sort_key)
                with open(file_path, "a") as f:
                    for entry in entries:
                        content = json.dumps(entry) + "\n"
                        f.write(content)
                        f.flush()

    #### Memory hook methods ####

    def _get_system_content(self, inference_data: dict) -> Optional[str]:
        """Return the current system message content from inference_data, or None."""
        for msg in inference_data.get("message", []):
            if msg.get("role") == "system":
                return msg["content"]
        return None

    def _replace_system_prompt_subtask(
        self,
        inference_data: dict,
        base_system_content: Optional[str],
        turn_instruction: str,
        test_entry: dict,
    ) -> dict:
        """
        Replace the subtask section of the system prompt with retrieval for the current turn.
        The workflow section (already in base_system_content) is kept intact.
        Thread-safe: uses local base_system_content, not instance state.
        """
        if self.memory_dir is None or "subtask" not in self.memory_types:
            return inference_data
        if not turn_instruction or base_system_content is None:
            return inference_data

        from bfcl_eval.memory.injection import get_subtask_text_for_turn
        subtask_text = get_subtask_text_for_turn(turn_instruction, test_entry, self.memory_dir)

        if subtask_text:
            subtask_block = f"\n\n## Subtask Memory (tool call examples)\n{subtask_text}"
            if "</memory>" in base_system_content:
                new_content = base_system_content.replace(
                    "</memory>", subtask_block + "\n</memory>", 1
                )
            else:
                new_content = base_system_content + "\n" + subtask_block
        else:
            new_content = base_system_content

        for msg in inference_data.get("message", []):
            if msg.get("role") == "system":
                msg["content"] = new_content
                break

        task_id = test_entry.get("id", "?")
        print(
            f"\n\033[1m\033[96m[SYS REPLACE] {task_id} | {turn_instruction[:60]}\033[0m",
            flush=True,
        )
        return inference_data

    def _apply_static_memory(self, test_entry: dict, override_types=None) -> dict:
        if self.memory_dir is None:
            return test_entry
        enabled = (override_types if override_types is not None else self.memory_types) & {"workflow", "subtask"}
        if not enabled:
            return test_entry
        from bfcl_eval.memory.injection import inject_static_memory
        return inject_static_memory(
            test_entry,
            self.memory_dir,
            decomposer_llm=self.decomposer_llm,
            enabled_types=enabled,
            subtask_source=self.subtask_source,
        )

    def _apply_dynamic_memory_for_turn(
        self,
        current_turn_message: list[dict],
        test_entry: dict,
    ) -> list[dict]:
        """Prepend per-turn retrieved memory to the user message for dynamic retrieval mode."""
        if self.memory_dir is None:
            return current_turn_message
        # Workflow is injected statically before Turn 1; dynamic injection is subtask-only.
        enabled = self.memory_types & {"subtask"}
        if not enabled:
            return current_turn_message

        turn_instruction = ""
        for msg in current_turn_message:
            if msg.get("role") == "user":
                turn_instruction = msg.get("content", "")
                break
        if not turn_instruction:
            return current_turn_message

        from bfcl_eval.memory.injection import inject_dynamic_memory_for_turn
        injection = inject_dynamic_memory_for_turn(
            turn_instruction,
            test_entry,
            self.memory_dir,
            enabled_types=enabled,
        )
        if not injection:
            return current_turn_message

        # Return a shallow copy with the user message content prepended
        result = []
        for msg in current_turn_message:
            if msg.get("role") == "user" and not msg.get("_memory_injected"):
                msg = {**msg, "content": injection + "\n\n" + msg["content"]}
            result.append(msg)
        return result

    def _augment_execution_results_with_hints(
        self,
        execution_results: list[str],
        decoded_model_responses: list,
        current_turn_message: list[dict],
        test_entry: dict = None,
    ) -> list[str]:
        if self.memory_dir is None or "function" not in self.memory_types:
            return execution_results
        from bfcl_eval.memory.injection import augment_with_function_hints
        turn_query = ""
        for msg in current_turn_message:
            if msg.get("role") == "user":
                turn_query = msg.get("content", "")
                break
        func_descs = {f["name"]: f for f in (test_entry or {}).get("function", []) if "name" in f}
        return augment_with_function_hints(
            execution_results,
            decoded_model_responses,
            self.memory_dir,
            turn_query=turn_query,
            func_descs=func_descs,
        )

    def _generate_pre_call_think(
        self,
        turn_idx: int,
        all_multi_turn_messages: list,
        all_model_response: list,
        current_turn_message: list,
        test_entry: dict,
    ) -> str:
        import re
        import sys

        current_instruction = ""
        for msg in current_turn_message:
            if msg.get("role") == "user":
                current_instruction = msg.get("content", "")
                break
        if not current_instruction:
            return ""

        prev_section = ""
        if turn_idx > 0:
            prev_lines = ["Previous turns:"]
            for i in range(turn_idx):
                turn_msgs = all_multi_turn_messages[i]
                for msg in turn_msgs:
                    if msg.get("role") == "user":
                        prev_lines.append(f"  Turn {i + 1} instruction: {msg.get('content', '')}")
                        break
                if i < len(all_model_response):
                    responses = all_model_response[i]
                    if responses:
                        prev_lines.append(f"  Turn {i + 1} tool calls: {responses}")
            prev_section = "\n".join(prev_lines) + "\n\n"

        tool_names = ", ".join(
            f.get("name", "") for f in test_entry.get("function", []) if f.get("name")
        )
        prompt = _THINK_USER_TEMPLATE.format(
            prev_section=prev_section,
            current_instruction=current_instruction,
            tool_names=tool_names or "unknown",
        )
        try:
            import os
            import litellm
            kwargs: dict = {}
            if self.think_model.startswith("openai/"):
                base_url = os.getenv("VLLM_BASE_URL", "")
                if base_url:
                    kwargs["api_base"] = base_url
                    kwargs["api_key"] = os.getenv("VLLM_API_KEY", "EMPTY")
            resp = litellm.completion(
                model=self.think_model,
                messages=[
                    {"role": "system", "content": _THINK_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                timeout=30.0,
                **kwargs,
            )
            content = resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"[think] error: {exc}", file=sys.stderr)
            return ""

        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
        return content

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        """
        Call the model API in FC mode to get the response.
        Return the response object that can be used to feed into the `_parse_query_response_FC` method.
        """
        raise NotImplementedError

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        This might includes transforming the input user message into the format expected by the model, extract out the system prompt (if any), and any other necessary preprocessing steps. Those steps can also be done in the `add_first_turn_message_FC` and `_add_next_turn_user_message_FC` methods, but it's usually cleaner to do it here.
        The inference_data dict is updated in place and returned.

        Note: This method has different signature from its Prompting version.
        """
        raise NotImplementedError

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """
        [Only for FC mode]
        This method is used to prepare/compile the tools from the test entry and add them to the inference data to use for model query in FC mode.
        Function docs usually need to be transformed to the format expected by the model, done through the `convert_to_tool` function from `model_handler/utils.py`.
        The inference_data dict is updated in place and returned.
        """
        raise NotImplementedError

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - tool_call_ids (list[str]): The IDs of the tool calls that are generated by the model. Optional.
                - Any other metadata that is specific to the model.
        """
        raise NotImplementedError

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history, in the format that the model expects.

        Args:
            inference_data (dict): The inference data from previous processing steps.
            first_turn_message (list[dict]): The first turn message from the test entry. It has variable length. It might contain one or more of the following roles:
                - "system": The system message. This role will only appear at most once, at the beginning of the first turn. For most entry, this role will not appear.
                - "user": The user message.
                - "assistant": The assistant message. For most entry, this role will not appear.

        Returns:
            inference_data (dict): The updated inference data that will be send to `_query_FC` to call the model API.
        """
        raise NotImplementedError

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next turn user message to the chat history for query.
        user_message is a list of 1 element, which is guaranteed to be a `user` role message.
        """
        raise NotImplementedError

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """
        raise NotImplementedError

    def _add_execution_results_FC(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        Some models may need to add additional information to the chat history, such as tool call IDs.
        """
        raise NotImplementedError

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        """
        Call the model API in prompting mode to get the response.
        Return the response object that can be used to feed into the decode method.
        """
        raise NotImplementedError

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        This might includes transforming the input user message into the format expected by the model, extract out the system prompt (if any), and any other necessary preprocessing steps. Those steps can also be done in the `add_first_turn_message_prompting` and `_add_next_turn_user_message_prompting` methods, but it's usually cleaner to do it here.
        The function docs are usually supplied to the prompting models as part of the system prompt, done via the `system_prompt_pre_processing_chat_model` function from `model_handler/utils.py`, unless the model has a different way of handling it.
        Returns a dict that contains all the necessary information for the query method.
        Things like `system_prompt` and `chat_history` are optional, specific to the model.

        Note: This method has different signature from its FC version.
        """
        raise NotImplementedError

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - Any other metadata that is specific to the model.
        """
        raise NotImplementedError

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history, in the format that the model expects.

        Args:
            inference_data (dict): The inference data from previous processing steps.
            first_turn_message (list[dict]): The first turn message from the test entry. It has variable length. It might contain one or more of the following roles:
                - "system": The system message. This role will only appear at most once, at the beginning of the first turn.
                - "user": The user message.
                - "assistant": The assistant message. For most entry, this role will not appear.

        Returns:
            inference_data (dict): The updated inference data that will be send to `_query_prompting` to call the model API.
        """
        raise NotImplementedError

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next turn user message to the chat history for query.
        user_message is a list of 1 element, which is guaranteed to be a `user` role message.
        """
        raise NotImplementedError

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """
        raise NotImplementedError

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        By default, execution results are added back as a `user` role message, as most models don't support the `tool` role in prompting mode.
        """
        raise NotImplementedError
