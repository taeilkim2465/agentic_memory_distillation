// SASM Teacher Agent Configuration (Adaptation Phase)
//
// Pure SASM: no playbook, no reflector, no curator.
// The teacher (large model) solves tasks and builds the SASM memory bank.
// After each task, the trajectory is decomposed into subtask segments and
// transferable (z, d, e) triples are stored in sasm_memory.json.

local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";

// Teacher model — large, high-capability
local teacher_model_config = {
    "name": "gpt-5-mini",
    "provider": "openai",
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
            "type": "sasm_adaptation_react",
            "generator_model_config": teacher_model_config,
            "extractor_model_config": teacher_model_config,
            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "generator_prompt_file_path": experiment_prompts_path + "/sasm_generator_prompt.txt",
            "sasm_memory_file_path": experiment_playbooks_path + "/sasm_memory.json",
            "sasm_decomposer_prompt_file_path": experiment_prompts_path + "/sasm_decomposer_prompt.txt",
            "sasm_extractor_prompt_file_path": experiment_prompts_path + "/sasm_extractor_prompt.txt",
            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "use_gt_code": true,
        },
        "dataset": "train",
    }
}
