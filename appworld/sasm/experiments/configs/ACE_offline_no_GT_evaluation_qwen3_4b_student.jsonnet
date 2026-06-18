local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_reasoning_bank_path = project_home_path + "/experiments/reasoning_bank";

local generator_model_config = {
    "name": "qwen3-4b",
    "provider": "vllm",
    "base_url": "ENDPOINT_HERE",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "logprobs": false,
    "top_logprobs": null,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": false,
    "max_retries": 50,
};

{
    "type": "ace",
    "config": {
        "run_type": "ace-evaluation",
        "agent": {
            "type": "ace_evaluation_react_bank",
            "generator_model_config": generator_model_config,
            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "generator_prompt_file_path": experiment_prompts_path + "/appworld_react_generator_prompt.txt",
            "reasoning_bank_file_path": experiment_reasoning_bank_path + "/offline_no_gt_gpt5mini_bank.json",
            "ignore_multiple_calls": true,
            "max_steps": 40,
            "top_k": 3,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
        },
        "dataset": "test_normal",
    }
}
