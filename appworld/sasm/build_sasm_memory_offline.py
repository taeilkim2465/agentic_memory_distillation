#!/usr/bin/env python3
"""
Offline SASM memory builder.

기존 실험 출력(lm_calls.jsonl + evaluation)을 읽어 SASM 메모리를 생성합니다.
teacher를 다시 실행하지 않고, memp/rb 등의 기존 trajectory를 재활용합니다.

Usage:
    PYTHONPATH=/c2/taeil/ace-appworld/sasm python sasm/build_sasm_memory_offline.py \
        --outputs-dir memp/experiments/outputs/MEMP_teacher_no_GT \
        --memory-path sasm/experiments/playbooks/sasm_memory.json \
        [--model gpt-4o-mini] \
        [--only-success] \
        [--num-processes 4]

    # rb 결과 사용
    PYTHONPATH=/c2/taeil/ace-appworld/sasm python sasm/build_sasm_memory_offline.py \
        --outputs-dir rb/experiments/outputs/ACE_offline_no_GT_adaptation_gpt5mini_teacher \
        --memory-path sasm/experiments/playbooks/sasm_memory_from_rb.json
"""

import argparse
import json
import os
import sys
from pathlib import Path


def get_task_dirs(outputs_dir: Path) -> list[Path]:
    tasks_root = outputs_dir / "tasks"
    if not tasks_root.exists():
        raise FileNotFoundError(f"tasks/ not found in {outputs_dir}")
    return sorted(
        d for d in tasks_root.iterdir()
        if d.is_dir() and (d / "logs" / "lm_calls.jsonl").exists()
    )


def read_lm_calls(task_dir: Path) -> list[dict]:
    path = task_dir / "logs" / "lm_calls.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def reconstruct_messages(lm_calls: list[dict]) -> list[dict]:
    """마지막 lm_call의 input.messages + output assistant 메시지로 전체 trajectory 재구성."""
    if not lm_calls:
        return []
    last = lm_calls[-1]
    messages = list(last["input"]["messages"])
    choices = last.get("output", {}).get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        if content:
            messages.append({"role": "assistant", "content": content})
    return messages


def get_task_success(task_dir: Path) -> bool:
    report = task_dir / "evaluation" / "report.md"
    if not report.exists():
        return False
    for line in report.read_text().splitlines():
        if "Num Failed Tests" in line:
            parts = line.split(":")
            if len(parts) == 2:
                return int(parts[1].strip()) == 0
    return False


def extract_task_instruction(messages: list[dict]) -> str:
    """첫 user 메시지 끝부분의 'Task: ...' 패턴에서 task instruction 추출."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            idx = content.rfind("Task:")
            if idx >= 0:
                return content[idx + len("Task:"):].strip().split("\n")[0].strip()
            return content[:300]
    return ""


def build_trajectory_text(messages: list[dict]) -> str:
    lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"[Step {i}] {role}:\n{content}\n")
    return "\n".join(lines)


def run_decomposer(model, decomposer_prompt: str, task_instruction: str, trajectory_text: str) -> list[dict]:
    from appworld_experiments.code.ace.playbook import extract_json_from_text
    prompt = (
        decomposer_prompt
        .replace("{{task_instruction}}", task_instruction)
        .replace("{{trajectory}}", trajectory_text)
    )
    response = model.generate(messages=[{"role": "user", "content": prompt}])
    parsed = extract_json_from_text(response.get("content", ""))
    if not parsed or "subtasks" not in parsed:
        return []
    return parsed["subtasks"]


def run_extractor(model, extractor_prompt: str, z: str, d: str,
                  messages: list[dict], start: int, end: int, success: bool) -> str | None:
    from appworld_experiments.code.ace.playbook import extract_json_from_text
    segment = messages[start: end + 1]
    segment_text = "\n".join(
        f"{m.get('role','').upper()}:\n{m.get('content','')}" for m in segment
    )
    prompt = (
        extractor_prompt
        .replace("{{z}}", z)
        .replace("{{d}}", d)
        .replace("{{trajectory_segment}}", segment_text)
        .replace("{{task_succeeded}}", "true" if success else "false")
    )
    response = model.generate(messages=[{"role": "user", "content": prompt}])
    parsed = extract_json_from_text(response.get("content", ""))
    if not parsed or "e" not in parsed:
        return None
    return parsed["e"]


def process_task(task_dir: Path, decomposer_prompt: str, extractor_prompt: str,
                 model_config: dict, only_success: bool) -> list[dict]:
    """단일 task를 처리해 (z, d, e) 트리플 리스트 반환."""
    from appworld_experiments.code.ace.lite_llm_generator import LiteLLMGenerator

    lm_calls = read_lm_calls(task_dir)
    if not lm_calls:
        return []

    messages = reconstruct_messages(lm_calls)
    success = get_task_success(task_dir)
    task_instruction = extract_task_instruction(messages)

    if only_success and not success:
        return []

    if not task_instruction:
        print(f"  [SKIP] {task_dir.name}: task instruction 없음")
        return []

    model = LiteLLMGenerator(**model_config)
    trajectory_text = build_trajectory_text(messages)

    subtasks = run_decomposer(model, decomposer_prompt, task_instruction, trajectory_text)
    if not subtasks:
        print(f"  [SKIP] {task_dir.name}: decomposer 결과 없음")
        return []

    triples = []
    for sub in subtasks:
        z = sub.get("z", "")
        d = sub.get("d", "")
        start = sub.get("start_step", 0)
        end = sub.get("end_step", len(messages) - 1)
        if not z or not d:
            continue
        e = run_extractor(model, extractor_prompt, z, d, messages, start, end, success)
        if e:
            triples.append({"z": z, "d": d, "e": e})

    outcome = "SUCCESS" if success else "FAILURE"
    print(f"  [{outcome}] {task_dir.name}: {len(triples)} entries 생성")
    return triples


def main():
    parser = argparse.ArgumentParser(description="Offline SASM memory builder")
    parser.add_argument("--outputs-dir", required=True,
                        help="기존 실험 출력 디렉토리 (tasks/ 포함)")
    parser.add_argument("--memory-path", required=True,
                        help="저장할 sasm_memory.json 경로")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="사용할 LLM 모델 (default: gpt-4o-mini)")
    parser.add_argument("--provider", default="openai",
                        help="모델 provider (default: openai)")
    parser.add_argument("--only-success", action="store_true",
                        help="성공한 task만 처리")
    parser.add_argument("--num-threads", type=int, default=1,
                        help="병렬 처리 스레드 수 (default: 1)")
    parser.add_argument("--clear", action="store_true",
                        help="기존 메모리 파일을 지우고 새로 시작")
    parser.add_argument("--temperature", type=float, default=0,
                        help="LLM temperature (default: 0, use 1 for models that don't support 0)")
    args = parser.parse_args()

    base = Path(__file__).parent
    decomposer_prompt = (base / "experiments/prompts/sasm_decomposer_prompt.txt").read_text()
    extractor_prompt = (base / "experiments/prompts/sasm_extractor_prompt.txt").read_text()

    outputs_dir = Path(args.outputs_dir)
    memory_path = Path(args.memory_path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    if args.clear and memory_path.exists():
        memory_path.unlink()
        print(f"기존 메모리 삭제: {memory_path}")

    model_config = {
        "name": args.model,
        "provider": args.provider,
        "temperature": args.temperature,
        "n": 1,
        "use_cache": True,
        "max_retries": 10,
        "retry_after_n_seconds": 10,
    }

    task_dirs = get_task_dirs(outputs_dir)
    print(f"처리할 task 수: {len(task_dirs)} (outputs: {outputs_dir})")
    print(f"저장 경로: {memory_path}")
    print(f"모델: {args.model} | only_success: {args.only_success} | threads: {args.num_threads}")
    print()

    from appworld_experiments.code.ace.sasm_memory import SASMMemoryBank
    memory = SASMMemoryBank(str(memory_path))

    all_triples = []
    if args.num_threads > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        futures = {}
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            for td in task_dirs:
                f = executor.submit(
                    process_task, td, decomposer_prompt, extractor_prompt,
                    model_config, args.only_success
                )
                futures[f] = td

        for f in as_completed(futures):
            try:
                all_triples.extend(f.result())
            except Exception as ex:
                print(f"  [ERROR] {futures[f].name}: {ex}")
    else:
        for i, td in enumerate(task_dirs):
            print(f"[{i+1}/{len(task_dirs)}] {td.name}")
            try:
                triples = process_task(td, decomposer_prompt, extractor_prompt,
                                       model_config, args.only_success)
                all_triples.extend(triples)
            except Exception as ex:
                print(f"  [ERROR] {td.name}: {ex}")

    for t in all_triples:
        memory.add(z=t["z"], d=t["d"], e=t["e"])

    print(f"\n완료: 총 {len(all_triples)}개 entries → {memory_path}")
    print(f"카테고리별: {memory.stats()}")


if __name__ == "__main__":
    main()
