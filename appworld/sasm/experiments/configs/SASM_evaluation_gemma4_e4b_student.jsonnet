// SASM Student — gemma4-e4b via vLLM
// Prerequisites: sasm_memory.json must exist in experiments/playbooks/

local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";

local student_model_config = {
    "name": "gemma4-e4b",
    "provider": "vllm",
    "base_url": "ENDPOINT_HERE",
    "temperature": 0,
    "seed": 100,
    "stop": ["<end_of_turn>"],
    "logprobs": false,
    "top_logprobs": null,
    "n": 1,
    "response_format": {"type": "text"},
    "max_tokens": 2048,
    "retry_after_n_seconds": 10,
    "use_cache": false,
    "max_retries": 50,
};

{
    "type": "ace",
    "config": {
        "run_type": "ace-evaluation",
        "agent": {
            "type": "sasm_evaluation_react",
            "generator_model_config": student_model_config,
            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "generator_prompt_file_path": experiment_prompts_path + "/sasm_generator_prompt.txt",
            "sasm_memory_file_path": experiment_playbooks_path + "/sasm_memory.json",
            "sasm_predictor_prompt_file_path": experiment_prompts_path + "/sasm_predictor_prompt.txt",
            "ignore_multiple_calls": true,
            "max_model_len": 131072,
            "max_output_tokens": 2048,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
        },
        "dataset": "test_normal",
    }
}
