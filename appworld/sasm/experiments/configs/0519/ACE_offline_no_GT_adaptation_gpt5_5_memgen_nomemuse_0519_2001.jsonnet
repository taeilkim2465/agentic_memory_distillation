local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local experiment_configs_path = project_home_path + "/experiments/configs";
local experiment_code_path = project_home_path + "/experiments/code";

local generator_model_config = {
    "name": "gpt-5.5",
    "provider": "openai",
    "base_url": null,
    "api_key": null,
    "temperature": 1,
    "seed": 100,
    "stop": null,
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

local reflector_model_config = {
    "name": "gpt-5.5",
    "provider": "openai",
    "base_url": null,
    "api_key": null,
    "temperature": 1,
    "seed": 100,
    "stop": null,
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

local curator_model_config = {
    "name": "gpt-5.5",
    "provider": "openai",
    "base_url": null,
    "api_key": null,
    "temperature": 1,
    "seed": 100,
    "stop": null,
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

{
    "type": "ace",
    "config": {
        "run_type": "ace-adaptation",
        "agent": {
            "type": "ace_adaptation_react",
            "generator_model_config": generator_model_config,
            "reflector_model_config": reflector_model_config,
            "curator_model_config": curator_model_config,
            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "use_reflector": false,
            "generator_prompt_file_path": experiment_prompts_path + "/appworld_react_generator_prompt.txt",
            "reflector_prompt_file_path": experiment_prompts_path + "/appworld_react_reflector_no_gt_prompt.txt",
            "curator_prompt_file_path": experiment_prompts_path + "/appworld_react_curator_prompt.txt",
            "retrieval_memory_build_prompt_file_path": experiment_prompts_path + "/appworld_retrieval_memory_build_prompt.txt",
            "initial_playbook_file_path": experiment_playbooks_path + "/appworld_initial_playbook.txt",
            "trained_playbook_file_path": experiment_playbooks_path + "/unused_playbook_when_retrieval_enabled.txt",
            "use_retrieval_memory": true,
            "use_retrieval_memory_for_prompt": false,
            "use_retrieval_memory_write": true,
            "retrieval_memory_file_path": experiment_playbooks_path + "/ace_retrieval_memory_gpt5_5_memgen_nomemuse_{experiment_name}.jsonl",
            "retrieval_top_k": 3,
            "retrieval_min_similarity": 0.30,
            "retrieval_max_chars": 4000,
            "retrieval_embedding_model": "text-embedding-3-small",
            "retrieval_embedding_provider": "openai",
            "retrieval_embedding_base_url": null,
            "retrieval_embedding_api_key": null,
            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "use_gt_code": false
        },
        "dataset": "test_normal",
        "num_epochs": 1,
    }
}
