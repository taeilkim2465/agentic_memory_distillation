// SASM Full Pipeline: Teacher → Student (same test_normal dataset)
//
// run_type "sasm" runs two phases in sequence:
//   Phase 1 (adaptation): Teacher solves test_normal tasks, builds sasm_memory.json
//   Phase 2 (evaluation): Student solves the same test_normal tasks using sasm_memory.json

local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";

local teacher_model_config = {
    "name": "gpt-5-mini",
    "provider": "openai",
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

local student_model_config = {
    "name": "gpt-4o-mini",
    "provider": "openai",
    "temperature": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

local sasm_memory_path = experiment_playbooks_path + "/sasm_memory.json";

{
    "type": "ace",
    "config": {
        "run_type": "sasm",

        "adaptation": {
            "agent": {
                "type": "sasm_adaptation_react",
                "generator_model_config": teacher_model_config,
                "extractor_model_config": teacher_model_config,
                "appworld_config": {"random_seed": 123},
                "logger_config": {"color": true, "verbose": true},
                "generator_prompt_file_path": experiment_prompts_path + "/sasm_generator_prompt.txt",
                "sasm_memory_file_path": sasm_memory_path,
                "sasm_decomposer_prompt_file_path": experiment_prompts_path + "/sasm_decomposer_prompt.txt",
                "sasm_extractor_prompt_file_path": experiment_prompts_path + "/sasm_extractor_prompt.txt",
                "ignore_multiple_calls": true,
                "max_steps": 40,
                "max_cost_overall": 1000,
                "max_cost_per_task": 10,
                "log_lm_calls": true,
                "use_gt_code": false,
            },
            "dataset": "test_normal",
        },

        "evaluation": {
            "agent": {
                "type": "sasm_evaluation_react",
                "generator_model_config": student_model_config,
                "appworld_config": {"random_seed": 123},
                "logger_config": {"color": true, "verbose": true},
                "generator_prompt_file_path": experiment_prompts_path + "/sasm_generator_prompt.txt",
                "sasm_memory_file_path": sasm_memory_path,
                "sasm_predictor_prompt_file_path": experiment_prompts_path + "/sasm_predictor_prompt.txt",
                "ignore_multiple_calls": true,
                "max_steps": 40,
                "max_cost_overall": 1000,
                "max_cost_per_task": 10,
                "log_lm_calls": true,
            },
            "dataset": "test_normal",  // same dataset as adaptation
        },
    }
}
