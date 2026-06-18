local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";

local model_config = {
    "name": "qwen3-4b",
    "provider": "openai",
    "base_url": "http://10.10.0.118:8001/v1",
    "api_key": "EMPTY",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "max_tokens": 4096,
    "retry_after_n_seconds": 10,
    "use_cache": false,
    "max_retries": 50,
    "chat_template_kwargs": {"enable_thinking": false},
};

local thinker_model_config = {
    [k]: model_config[k]
    for k in std.objectFields(model_config)
    if k != "chat_template_kwargs"
} + {
    "chat_template_kwargs": {"enable_thinking": false},
};

{
    "type": "ace",
    "config": {
        "run_type": "ace-adaptation",
        "agent": {
            "type": "running_memory_react",
            "generator_model_config": model_config,
            "reflector_model_config": model_config,
            "curator_model_config": model_config,
            "thinker_model_config": thinker_model_config,
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
            "initial_playbook_file_path": experiment_playbooks_path + "/appworld_initial_playbook.txt",
            "trained_playbook_file_path": experiment_playbooks_path + "/unused_playbook_when_retrieval_enabled.txt",
            "function_memory_graph_path": project_home_path + "/experiments/outputs/ACE_offline_no_GT_adaptation_gpt5mini_memgen_nomemuse_0323_0931/function_memory_graph_perfect_v4.json",
            "graph_outgoing_candidate_k": 0,
            "function_memory_file_path": experiment_playbooks_path + "/ace_retrieval_memory_gpt5mini_memgen_nomemuse_ACE_offline_no_GT_adaptation_gpt5mini_memgen_nomemuse_0323_0931",
            "function_memory_top_k": 1,
            "graph_outgoing_top_k": 3,
            "retrieval_memory_file_path": experiment_playbooks_path + "/ace_retrieval_memory_gpt5mini_memgen_nomemuse_ACE_offline_no_GT_adaptation_gpt5mini_memgen_nomemuse_0323_0931",
            "retrieval_top_k": 1,
            "retrieval_min_similarity": 0.30,
            "retrieval_max_chars": 4000,
            "retrieval_embedding_model": "text-embedding-3-small",
            "retrieval_embedding_provider": "openai",
            "retrieval_memory_build_prompt_file_path": experiment_prompts_path + "/appworld_retrieval_memory_build_prompt.txt",
            "ignore_multiple_calls": true,
            "max_prompt_length": 60000,
            "max_output_length": 50000,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "use_gt_code": false,
            // Running Memory 설정
            "use_running_memory": true,
            "context_max_keys": 5,
        },
        "dataset": "test_normal",
        "num_epochs": 1,
    }
}
