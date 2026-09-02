# benchmark_runner.py
# CSE445 Assignment 3 - Mashrafe Bin Morshed
#
# Task 3 benchmark: runs the agent across a grid of datasets x algorithms,
# double-checks the numbers by recomputing them straight from the tool
# calls (not trusting whatever markdown the LLM writes), and saves a report.

import argparse
from datetime import datetime

from react_agent import OllamaBackend, MockBackend, run_agent_loop, MODEL_NAME

DEFAULT_DATASETS = ["wine", "breast_cancer"]
DEFAULT_ALGORITHMS = ["decision_tree", "logistic_regression", "random_forest"]


def build_benchmark_prompt(datasets, algorithms):
    pairs = ", ".join(f"{d}/{a}" for d in datasets for a in algorithms)
    n_calls = len(datasets) * len(algorithms)
    return (
        f"Run train_sklearn_model on every combination of these datasets and algorithms: {pairs}. "
        f"That is {n_calls} tool calls total, one per combination - you must actually call the tool "
        f"{n_calls} times, you cannot answer from memory or make up numbers. "
        f"Use ONLY train_sklearn_model for this - do not call tune_hyperparameters or any other tool, "
        f"they will not count toward the {n_calls} required combinations. "
        f"Start now with the first combination: Action: train_sklearn_model. "
        f"Only after you have {n_calls} real Observations should you give a Final Answer with a "
        f"markdown table of dataset, algorithm, test_accuracy, and cv_mean_accuracy."
    )


def build_missing_hint_fn(datasets, algorithms, prior_observations=None):
    """Returns a callback that is the authority on whether the benchmark's exact grid of
    required train_sklearn_model calls is done - returns None once every combination has a
    real result, otherwise a hint string naming exactly what's still missing plus a
    ready-to-copy Action Input for the next one (a small model is much more likely to follow
    that than to track the remaining grid on its own).

    This only counts calls that actually match a (dataset, algorithm) combo in the grid, so a
    model that calls a different tool (e.g. tune_hyperparameters) instead of train_sklearn_model
    does not get credited as progress - it's still missing everything it hasn't actually run.

    prior_observations: real results already collected from EARLIER attempts (see the
    outer retry loop in main()) - combined with the current run's own observations so a
    fresh attempt correctly recognizes work already done in a previous attempt instead of
    demanding it be redone."""
    required = [(d, a) for d in datasets for a in algorithms]
    prior_observations = prior_observations or []

    def hint_fn(real_observations):
        combined = prior_observations + real_observations
        done = {
            (o.get("dataset_name"), o.get("model_type"))
            for o in combined
            if "test_accuracy" in o
        }
        missing = [(d, a) for d, a in required if (d, a) not in done]
        if not missing:
            return None
        next_d, next_a = missing[0]
        missing_list = ", ".join(f"{d}/{a}" for d, a in missing)
        return (
            f"Still missing: {missing_list}. Call train_sklearn_model next with "
            f'Action Input: {{"dataset_name": "{next_d}", "model_type": "{next_a}"}}'
        )

    return hint_fn


def build_continuation_prompt(missing_combos):
    """A fresh, short prompt for a retry attempt - only mentions what's still missing,
    instead of restating the whole original grid, so it doesn't get any longer/slower to
    process just because earlier attempts already made some progress."""
    pairs = ", ".join(f"{d}/{a}" for d, a in missing_combos)
    n = len(missing_combos)
    return (
        f"Continuing a benchmark. The following combinations are already done and must NOT "
        f"be repeated - you still need to run train_sklearn_model on exactly these {n} "
        f"remaining combination(s): {pairs}. "
        f"Start now with the first one: Action: train_sklearn_model. "
        f"Only after all {n} are done should you give a Final Answer."
    )


def ground_truth_table(real_observations, datasets, algorithms):
    """Recompute the results table directly from the dicts tools actually returned - never
    by re-parsing transcript text, since a small model will sometimes type text that looks
    like 'Observation: {...}' inside its own response without the tool having run at all."""
    rows = {}
    for data in real_observations:
        if "test_accuracy" not in data:
            continue
        key = (data.get("dataset_name"), data.get("model_type"))
        rows[key] = data

    table = []
    missing = []
    for d in datasets:
        for a in algorithms:
            if (d, a) in rows:
                r = rows[(d, a)]
                table.append((d, a, r["test_accuracy"], r["cv_mean_accuracy"]))
            else:
                missing.append((d, a))
    return table, missing


def render_markdown_table(table):
    lines = ["| Dataset | Algorithm | Test Accuracy | CV Mean Accuracy |", "|---|---|---|---|"]
    for d, a, test_acc, cv_acc in table:
        lines.append(f"| {d} | {a} | {test_acc} | {cv_acc} |")
    return "\n".join(lines)


def run_benchmark_with_retries(backend, datasets, algorithms, max_iterations, max_attempts, verbose=True):
    """Runs the benchmark agent, and if a single attempt doesn't complete the full
    dataset x algorithm grid, automatically starts a fresh attempt asking ONLY for what's
    still missing - up to max_attempts total - instead of requiring the whole grid to be
    completed in one unbroken run. Every real result from every attempt is kept (the
    underlying sklearn computation is deterministic, so results are consistent whichever
    attempt produced them); this is a higher-level, task-scale version of the same
    self-correction principle as the per-step guards inside run_agent_loop itself.

    Returns (all_real_observations, attempt_summaries) where attempt_summaries is a list of
    dicts with the log/final_answer/forced_stop for each attempt, for the report."""
    all_real_observations = []
    attempt_summaries = []

    for attempt in range(1, max_attempts + 1):
        _, missing = ground_truth_table(all_real_observations, datasets, algorithms)
        if not missing:
            break

        if attempt == 1:
            prompt = build_benchmark_prompt(datasets, algorithms)
        else:
            prompt = build_continuation_prompt(missing)

        hint_fn = build_missing_hint_fn(datasets, algorithms, prior_observations=all_real_observations)

        print(f"\n########## Benchmark attempt {attempt}/{max_attempts} "
              f"({len(missing)} combination(s) still needed) ##########\n")
        result = run_agent_loop(
            prompt, backend, max_iterations=max_iterations, verbose=verbose,
            min_tool_calls=len(missing), missing_hint_fn=hint_fn,
        )
        all_real_observations.extend(result["real_observations"])
        attempt_summaries.append({
            "attempt": attempt,
            "log": result["log"],
            "final_answer": result["final_answer"],
            "forced_stop": result["forced_stop"],
        })

    return all_real_observations, attempt_summaries


def main():
    parser = argparse.ArgumentParser(description="Run the Task 3 benchmark and write a report.")
    parser.add_argument("--backend", choices=["ollama", "mock"], default="ollama")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=4,
                         help="If the agent doesn't finish the full grid in one run, automatically "
                              "retry with a follow-up prompt covering only what's missing, up to "
                              "this many attempts total.")
    parser.add_argument("--output", type=str, default="benchmark_report.md")
    parser.add_argument("--timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    args = parser.parse_args()

    backend = OllamaBackend(model=args.model, timeout=args.timeout) if args.backend == "ollama" else MockBackend()
    n_calls = len(args.datasets) * len(args.algorithms)

    print(f"Running benchmark across {args.datasets} x {args.algorithms} "
          f"({n_calls} required calls, up to {args.max_attempts} attempt(s))...\n")
    all_real_observations, attempt_summaries = run_benchmark_with_retries(
        backend, args.datasets, args.algorithms, args.max_iterations, args.max_attempts,
    )

    table, missing = ground_truth_table(all_real_observations, args.datasets, args.algorithms)
    md_table = render_markdown_table(table)

    best = max(table, key=lambda row: row[2]) if table else None

    report_lines = [
        "# Benchmark Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Backend: {type(backend).__name__}, model: {args.model if args.backend == 'ollama' else 'n/a'}",
        f"Attempts used: {len(attempt_summaries)}/{args.max_attempts} "
        "(a new attempt only starts if the previous one left combinations unfinished; "
        "each one is told exactly what's already done so it never repeats work)",
        "",
        "## Results (recomputed from tool observations, not the model's own summary)",
        "",
        md_table,
        "",
    ]

    if missing:
        report_lines.append(f"**Missing combinations the agent never completed in {args.max_attempts} attempt(s):** {missing}")
        report_lines.append("")
    else:
        report_lines.append(f"**All {n_calls} required combinations completed.**")
        report_lines.append("")

    if best:
        report_lines.append(f"**Best combination:** {best[0]}/{best[1]} with test accuracy {best[2]}")
        report_lines.append("")

    for summary in attempt_summaries:
        report_lines.append(f"## Attempt {summary['attempt']} - Agent's own Final Answer")
        report_lines.append("")
        report_lines.append(summary["final_answer"] or "(none)")
        report_lines.append("")
        report_lines.append(f"## Attempt {summary['attempt']} - Full execution trace")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append(summary["log"])
        report_lines.append("```")
        report_lines.append("")

    with open(args.output, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
