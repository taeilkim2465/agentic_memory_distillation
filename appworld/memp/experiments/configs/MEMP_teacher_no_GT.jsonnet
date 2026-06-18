// Mem^p Teacher — solves test_normal, grades each task, and builds memory.
//
// After all tasks: experiments/memory/memp_teacher_store.json contains
// the accumulated memory (trajectory + script per task).

local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local memp_memory_path        = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH") + "/../memory/memp_teacher_store.json";

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

            "generator_model_config":     base_model_config,
            "proceduralize_model_config": base_model_config,
            "adjust_model_config":        base_model_config,
            "keyword_model_config":       base_model_config,

            "memory_store_path": memp_memory_path,

            "appworld_config": { "random_seed": 123 },
            "logger_config":   { "color": true, "verbose": true },

            "generator_prompt_file_path":     experiment_prompts_path + "/memp_generator_prompt.txt",
            "proceduralize_prompt_file_path": experiment_prompts_path + "/memp_proceduralize_prompt.txt",
            "adjust_prompt_file_path":        experiment_prompts_path + "/memp_adjust_prompt.txt",
            "keyword_prompt_file_path":       experiment_prompts_path + "/memp_keyword_prompt.txt",

            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "retrieval_top_k": 3,
        },
        "dataset": "test_normal",
    }
}
