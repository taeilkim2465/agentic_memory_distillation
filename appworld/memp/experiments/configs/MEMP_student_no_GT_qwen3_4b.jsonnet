// Mem^p Student — qwen3-4b via vLLM
// Prerequisites: run MEMP_teacher_no_GT first.

local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local memp_memory_path        = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH") + "/../memory/memp_teacher_store.json";

local student_model_config = {
    "name": "qwen3-4b",
    "provider": "vllm",
    "base_url": "http://localhost:8881/v1",
    "api_key": "EMPTY",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "chat_template_kwargs": {"enable_thinking": false},
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
        "run_type": "memp-student",
        "agent": {
            "type": "memp_student_react",

            "generator_model_config": student_model_config,
            "keyword_model_config":   student_model_config,

            "memory_store_path": memp_memory_path,

            "appworld_config": { "random_seed": 123 },
            "logger_config":   { "color": true, "verbose": true },

            "generator_prompt_file_path": experiment_prompts_path + "/memp_generator_prompt.txt",
            "keyword_prompt_file_path":   experiment_prompts_path + "/memp_keyword_prompt.txt",

            "ignore_multiple_calls": true,
            "max_model_len": 40960,
            "max_output_tokens": 2048,
            "context_buffer": 1536,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "retrieval_top_k": 3,
            "retrieval_embedding_model": "text-embedding-3-small",
            "retrieval_embedding_base_url": "https://api.openai.com/v1",
        },
        "dataset": "test_normal",
    }
}
