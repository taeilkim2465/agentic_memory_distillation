// Mem^p online (no ground-truth) configuration.
//
// Memory lifecycle per task:
//   retrieve → execute → evaluate → (validate | adjust) → store
//
// All four model configs point to the same endpoint by default;
// swap proceduralize/adjust/keyword to a cheaper model if desired.

local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local memp_memory_path        = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH") + "/../memory/memp_store.json";

local base_model_config = {
    "name": "gpt-4o-mini",
    "provider": "openai",
    "temperature": 0,
    "seed": 100,
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": false,
    "max_retries": 50,
};

{
    "type": "ace",
    "config": {
        "run_type": "memp",
        "agent": {
            "type": "memp_react",

            // Generator: solves the task using retrieved memory context
            "generator_model_config": base_model_config,

            // Proceduralize: abstracts a trajectory into a reusable script
            "proceduralize_model_config": base_model_config,

            // Adjust: updates an existing script given a failure signal
            "adjust_model_config": base_model_config,

            // Keyword: extracts keywords from a task description for retrieval
            "keyword_model_config": base_model_config,

            // Path to the persistent JSON memory store
            "memory_store_path": memp_memory_path,

            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },

            // Prompt file paths
            "generator_prompt_file_path":      experiment_prompts_path + "/memp_generator_prompt.txt",
            "proceduralize_prompt_file_path":  experiment_prompts_path + "/memp_proceduralize_prompt.txt",
            "adjust_prompt_file_path":         experiment_prompts_path + "/memp_adjust_prompt.txt",
            "keyword_prompt_file_path":        experiment_prompts_path + "/memp_keyword_prompt.txt",

            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,

            // Number of memories to retrieve per task (AveFact top-k)
            "retrieval_top_k": 3,
        },
        "dataset": "test_normal",
    }
}
