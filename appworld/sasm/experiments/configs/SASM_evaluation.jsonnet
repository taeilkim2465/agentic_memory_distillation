// SASM Student Agent Configuration (Evaluation Phase)
//
// Pure SASM: no playbook, no reflector, no curator.
// The student (small model) uses the teacher-built SASM memory bank.
// At each step: predict subtask (z, d) → retrieve experience → inject → generate.

local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";

// Student model — can be smaller/cheaper than teacher
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

{
    "type": "ace",
    "config": {
        "run_type": "ace-evaluation",
        "agent": {
            "type": "sasm_evaluation_react",
            "generator_model_config": student_model_config,
            // predictor_model_config omitted: falls back to generator model
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
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
        },
        "dataset": "test_normal",
    }
}
