// Mem^p Student — solves test_normal using ONLY teacher's pre-built memory.
//
// Prerequisites: run MEMP_teacher_no_GT first so that
//   experiments/memory/memp_teacher_store.json exists.
//
// The student reads the store but never writes to it (read-only).
// The student model can be swapped to a weaker model to test
// cross-model memory transfer (Concept 5 of the Mem^p paper).

local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local memp_memory_path        = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH") + "/../memory/memp_teacher_store.json";

local student_model_config = {
    "name": "gpt-4o-mini",   // swap to a weaker model to test cross-model transfer
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
        "run_type": "memp-student",
        "agent": {
            "type": "memp_student_react",

            "generator_model_config": student_model_config,
            "keyword_model_config":   student_model_config,  // same or lighter model

            // Points to the teacher's store — read-only at runtime
            "memory_store_path": memp_memory_path,

            "appworld_config": { "random_seed": 123 },
            "logger_config":   { "color": true, "verbose": true },

            "generator_prompt_file_path": experiment_prompts_path + "/memp_generator_prompt.txt",
            "keyword_prompt_file_path":   experiment_prompts_path + "/memp_keyword_prompt.txt",

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
