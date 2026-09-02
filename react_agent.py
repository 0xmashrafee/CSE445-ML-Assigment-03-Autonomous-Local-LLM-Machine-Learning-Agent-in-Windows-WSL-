# react_agent.py
# CSE445 Assignment 3 - Mashrafe Bin Morshed
#
# A small ReAct-style agent loop. The LLM (local model through Ollama) writes
# Thought / Action / Action Input steps, we run the matching tool from
# ml_tools.py, feed the result back as an Observation, and repeat until the
# model gives a Final Answer.

import argparse
import json
import os
import re
import time
from datetime import datetime

import requests

from ml_tools import AVAILABLE_TOOLS, TOOL_DESCRIPTIONS

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

MAX_RETRIES_PER_TOOL = 3      # Task 3 self-healing budget
MAX_IDENTICAL_REPEATS = 3     # stop the agent if it keeps calling the same tool with the same args


def build_system_prompt():
    tool_lines = "\n".join(f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items())
    return f"""You are an autonomous ML agent. You solve the user's request by choosing tools and
reasoning step by step in this exact format:

Thought: <your reasoning>
Action: <tool name>
Action Input: <a single line of valid JSON with the tool's arguments>

After each Action Input, wait - you will be given an Observation with the tool's result.
Keep looping Thought/Action/Action Input/Observation until you have enough information,
then finish with:

Thought: <final reasoning>
Final Answer: <a clear answer for the user, summarizing the numbers you found>

Available tools:
{tool_lines}

Rules:
- Only call one tool per step.
- Action Input must be valid JSON on one line, e.g. {{"dataset_name": "iris", "model_type": "decision_tree"}}
- If a tool returns {{"error": ...}}, read the message, change your arguments to fix the problem, and try again.
- Never call the same tool with the exact same Action Input more than once - you already have that
  Observation, calling it again will not produce a new result. Move on once you have enough evidence.
- Do not invent numbers. Only report results that came from a tool Observation.
- You cannot give a Final Answer before you have called at least one tool. If the task asks for
  several combinations, call the tool once per combination before answering.
"""


SYSTEM_PROMPT = build_system_prompt()

ACTION_RE = re.compile(r"Action:\s*(.+)")
ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+)")
FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def parse_action(text):
    """Pull the last Action / Action Input pair out of a model response."""
    final_match = FINAL_RE.search(text)
    if final_match:
        return None, None, final_match.group(1).strip()

    action_matches = ACTION_RE.findall(text)
    input_matches = ACTION_INPUT_RE.findall(text)
    if not action_matches or not input_matches:
        return None, None, None

    tool_name = action_matches[-1].strip()
    raw_input = input_matches[-1].strip()
    # models sometimes wrap the json in backticks or add trailing text
    raw_input = raw_input.strip("`")
    try:
        kwargs = json.loads(raw_input)
    except json.JSONDecodeError:
        # try to grab just the {...} part
        match = re.search(r"\{.*\}", raw_input, re.DOTALL)
        if not match:
            return tool_name, {}, None
        try:
            kwargs = json.loads(match.group(0))
        except json.JSONDecodeError:
            return tool_name, {}, None
    return tool_name, kwargs, None


class OllamaBackend:
    """Talks to a real local model running under Ollama."""

    def __init__(self, model=MODEL_NAME, url=OLLAMA_URL, timeout=300):
        self.model = model
        self.url = url
        self.timeout = timeout
        self.last_metadata = {}

    def generate(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
            # stop generation the moment the model tries to write its own fake
            # Observation or start a second Thought - keeps it to one step at a time
            "stop": ["Observation:", "\nThought:"],
        }
        t0 = time.perf_counter()
        resp = requests.post(self.url, json=payload, timeout=self.timeout)
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()

        self.last_metadata = {
            "latency_s": round(elapsed, 3),
            "eval_count": data.get("eval_count"),
            "eval_duration_ns": data.get("eval_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
        }
        return data.get("response", "")


class MockBackend:
    """
    Offline, rule-based stand-in for the LLM. Used for testing the agent loop
    and generating demo traces on machines without Ollama installed. This is
    NOT a language model - it just pattern-matches keywords in the query to
    decide which tool to call next, so the reasoning it writes is scripted.
    """

    def __init__(self):
        self.last_metadata = {"note": "mock backend - no real model call"}
        self._plans = {}

    def _plan_for(self, user_query):
        if user_query not in self._plans:
            self._plans[user_query] = self._build_plan(user_query)
        return self._plans[user_query]

    def _build_plan(self, q):
        ql = q.lower()
        steps = []

        combo_pairs = re.findall(r"(wine|breast_cancer|iris)/(decision_tree|logistic_regression|random_forest)", ql)
        if combo_pairs:
            for dataset, model_type in combo_pairs:
                steps.append(("train_sklearn_model", {"dataset_name": dataset, "model_type": model_type}))
        elif "compare" in ql and ("decision tree" in ql or "logistic" in ql or "random forest" in ql):
            dataset = "wine" if "wine" in ql else ("breast_cancer" if "breast" in ql or "cancer" in ql else "iris")
            for model_type in ["decision_tree", "logistic_regression", "random_forest"]:
                steps.append(("train_sklearn_model", {"dataset_name": dataset, "model_type": model_type}))
        elif "grid search" in ql or "hyperparameter" in ql or "tune" in ql:
            dataset = "wine" if "wine" in ql else "iris"
            steps.append(("tune_hyperparameters", {"dataset_name": dataset, "model_type": "svc", "search_type": "grid"}))
        elif "pca" in ql or "dimensionality" in ql or "feature selection" in ql:
            dataset = "breast_cancer" if "breast" in ql or "cancer" in ql else "wine"
            steps.append(("reduce_dimensionality", {"dataset_name": dataset, "method": "pca", "n_components": 2}))
        elif "batch" in ql or "batchnorm" in ql or "shape" in ql:
            dataset = "wine"
            steps.append(("train_deep_pytorch_classifier", {"dataset_name": dataset, "batch_size": 1}))
            steps.append(("train_deep_pytorch_classifier", {"dataset_name": dataset, "batch_size": 16}))
        elif "nan" in ql or "diverge" in ql or "learning rate" in ql:
            dataset = "breast_cancer"
            steps.append(("train_pytorch_mlp", {"dataset_name": dataset, "lr": 50, "optimizer_name": "sgd"}))
            steps.append(("train_pytorch_mlp", {"dataset_name": dataset, "lr": 0.01, "optimizer_name": "sgd"}))
        else:
            dataset = "iris" if "iris" not in ql and "wine" not in ql and "cancer" not in ql else (
                "wine" if "wine" in ql else ("breast_cancer" if "cancer" in ql else "iris")
            )
            steps.append(("load_dataset_summary", {"dataset_name": dataset}))
            steps.append(("train_sklearn_model", {"dataset_name": dataset, "model_type": "decision_tree"}))

        return steps

    def generate(self, prompt):
        # figure out where we are by counting Observations already in the prompt
        user_query_match = re.search(r"User request: (.+)", prompt)
        user_query = user_query_match.group(1).strip() if user_query_match else prompt
        plan = self._plan_for(user_query)

        n_done = prompt.count("Observation:")
        if n_done >= len(plan):
            return f"Thought: I have run {n_done} tools and have enough results.\nFinal Answer: See the observations above for the numeric results."

        tool_name, kwargs = plan[n_done]
        return (
            f"Thought: I should run {tool_name} to make progress on this request.\n"
            f"Action: {tool_name}\n"
            f"Action Input: {json.dumps(kwargs)}"
        )


def run_agent_loop(user_query, backend, max_iterations=8, verbose=True, min_tool_calls=1, missing_hint_fn=None):
    """
    missing_hint_fn: optional callback(real_observations) -> str | None. When given, it is
    the authority on completion: return None once everything required is actually done, or
    a hint string naming exactly what's still missing otherwise. Overrides min_tool_calls'
    plain "count of any successful call" check, which a model can inflate for free by calling
    a different, unrequested tool. Used when the caller knows exactly which calls are still
    required (e.g. the benchmark's fixed dataset x algorithm grid) - a 3B model is far more
    likely to follow an exact copyable Action Input than to track a remaining grid on its own.
    """
    transcript = f"{SYSTEM_PROMPT}\n\nUser request: {user_query}\n"
    log_lines = [f"=== Agent run: {user_query} ===", f"backend={type(backend).__name__}", ""]

    failure_counts = {}   # tool_name -> consecutive error count, for self-healing
    repeat_counts = {}    # (tool_name, args) signature -> how many times called

    reached_final = False
    forced_stop = False
    final_answer = None
    n_tool_errors = 0
    n_successful_calls = 0     # real tool calls that returned a usable (non-error) result
    real_observations = []     # the actual dicts returned by successful tool calls only -
                                # never re-parsed from transcript text, since a small model
                                # will sometimes type text that looks like "Observation: {...}"
                                # inside its own response without actually calling the tool
    empty_final_attempts = 0   # times the model tried to answer before calling any tool
    # Worst case, a model that stalls needs one "reject + here's the exact next call" round
    # per remaining combination before it locks onto the pattern. That's still bounded and
    # always forward-progressing (each round either gets a real call or gives up), so scale
    # the retry budget to the task size instead of a flat constant - max_iterations is the
    # real backstop against a model that just won't cooperate at all.
    max_empty_attempts = max(3, min_tool_calls)

    for step in range(max_iterations):
        try:
            response = backend.generate(transcript)
        except Exception as e:
            # a slow/CPU-only model can blow past the request timeout, or the
            # Ollama connection can drop mid-run - don't let that kill the
            # whole process, just stop this one task with what we have so far
            log_lines.append(f"[step {step}] backend request failed: {e}")
            final_answer = _synthesize_final_answer(real_observations)
            if final_answer == "No usable tool results were produced.":
                final_answer = f"Stopped early - the model backend failed to respond ({e})."
            reached_final = True
            forced_stop = True
            break
        tool_name, kwargs, final = parse_action(response)

        if final is not None:
            # A small model will sometimes skip straight to a "Final Answer" full of
            # made-up numbers instead of actually calling every tool the task needs -
            # e.g. it calls the tool twice out of six required combinations, then
            # answers as if it had all six. Refuse a Final Answer until the required
            # work is actually done.
            #
            # "Done" is judged two different ways depending on what the caller told us:
            #   - if missing_hint_fn is given, it's the authority - it knows the exact
            #     grid of calls needed (e.g. the benchmark's dataset x algorithm combos)
            #     and returns None only once every one of them has a real result. This
            #     matters because n_successful_calls counts ANY successful tool call -
            #     a model that calls a different, unrequested tool (e.g. tune_hyperparameters
            #     instead of train_sklearn_model) still bumps that counter without making
            #     real progress on the grid, which would otherwise let it slip through
            #     early or throw off the "how many remain" arithmetic shown to the model.
            #   - otherwise, fall back to the simple min_tool_calls count.
            specific_hint = missing_hint_fn(real_observations) if missing_hint_fn is not None else None
            is_incomplete = (specific_hint is not None) if missing_hint_fn is not None else (n_successful_calls < min_tool_calls)

            if is_incomplete:
                empty_final_attempts += 1
                if empty_final_attempts <= max_empty_attempts:
                    if specific_hint:
                        detail = specific_hint
                    else:
                        remaining = min_tool_calls - n_successful_calls
                        detail = (f"You have no real numbers for the missing {remaining} - do not guess or "
                                  f"reuse a number from a different call. Call the tool again with the next "
                                  f"combination.")
                    hint = (
                        f"\nObservation: Your Final Answer was rejected - the required tool calls are not "
                        f"all done yet. {detail}\n"
                    )
                    transcript += response + hint
                    log_lines.append(f"[step {step}] rejected an incomplete Final Answer "
                                      f"({n_successful_calls} total successful call(s) so far, attempt {empty_final_attempts})")
                    continue
                # gave it several chances - stop rather than loop forever on a model that won't cooperate
                log_lines.append(f"[step {step}] model still hasn't completed the required calls - stopping")
                final_answer = _synthesize_final_answer(real_observations)
                if final_answer == "No usable tool results were produced.":
                    final_answer = "The model did not complete all required tool calls, so no real results are available."
                elif missing_hint_fn is not None:
                    # n_successful_calls may include off-grid calls to other tools, so an
                    # "X/Y completed" count here could be misleading - keep it generic
                    final_answer = (f"Not all required tool calls were completed. Real results for what "
                                     f"was actually run:\n{final_answer}")
                else:
                    final_answer = (f"Only {n_successful_calls}/{min_tool_calls} required tool calls were "
                                     f"completed. Real results for those:\n{final_answer}")
                reached_final = True
                forced_stop = True
                transcript += response
                break

            final_answer = final
            reached_final = True
            transcript += response
            log_lines.append(f"[step {step}] Final Answer: {final}")
            break

        if tool_name is None or tool_name not in AVAILABLE_TOOLS:
            transcript += response + f"\nObservation: '{tool_name}' is not a valid tool name. Valid tools: {list(AVAILABLE_TOOLS.keys())}\n"
            log_lines.append(f"[step {step}] invalid tool name: {tool_name}")
            continue

        sig = f"{tool_name}::{json.dumps(kwargs, sort_keys=True)}"
        repeat_counts[sig] = repeat_counts.get(sig, 0) + 1

        if repeat_counts[sig] >= MAX_IDENTICAL_REPEATS:
            log_lines.append(f"[step {step}] {tool_name} called identically {repeat_counts[sig]} times - forcing a stop")
            final_answer = _synthesize_final_answer(real_observations)
            transcript += response + f"\nFinal Answer: {final_answer}\n"
            reached_final = True
            forced_stop = True
            break

        if repeat_counts[sig] == MAX_IDENTICAL_REPEATS - 1:
            # same specific-next-step trick as the Final Answer guard: a vague "try
            # different arguments" often just gets echoed right back, since the model
            # has no way to know which arguments are actually still needed - if we can
            # name the exact missing combination, use that instead
            detail = "Try different arguments or give your Final Answer."
            if missing_hint_fn is not None:
                specific = missing_hint_fn(real_observations)
                if specific:
                    detail = specific
            hint = f"\nObservation: HINT - you already called {tool_name} with these exact arguments, that will not produce a new result. {detail}\n"
            transcript += response + hint
            log_lines.append(f"[step {step}] repeat warning for {tool_name}")
            continue

        try:
            result_json = AVAILABLE_TOOLS[tool_name](**kwargs)
            result = json.loads(result_json)
        except TypeError as e:
            # the model called a real tool with a missing/extra/misspelled argument -
            # treat this exactly like any other tool error instead of crashing the run
            result = {"error": f"Bad arguments for {tool_name}: {e}. Check the exact "
                                f"parameter names listed for this tool and try again."}
            result_json = json.dumps(result)
        except Exception as e:
            result = {"error": f"{tool_name} raised an unexpected error: {e}"}
            result_json = json.dumps(result)

        if "error" in result:
            n_tool_errors += 1
            failure_counts[tool_name] = failure_counts.get(tool_name, 0) + 1
            if failure_counts[tool_name] > MAX_RETRIES_PER_TOOL:
                transcript += response + f"\nObservation: {result_json}\nObservation: giving up on {tool_name} after {MAX_RETRIES_PER_TOOL} failed attempts.\n"
                log_lines.append(f"[step {step}] {tool_name} failed too many times, giving up on this tool")
                continue
            transcript += response + f"\nObservation: {result_json}\n"
            log_lines.append(f"[step {step}] {tool_name}({kwargs}) -> ERROR: {result.get('error')}")
        else:
            failure_counts[tool_name] = 0
            n_successful_calls += 1
            real_observations.append(result)
            transcript += response + f"\nObservation: {result_json}\n"
            log_lines.append(f"[step {step}] {tool_name}({kwargs}) -> {result_json}")

    if not reached_final:
        final_answer = _synthesize_final_answer(real_observations)
        transcript += f"\nFinal Answer: {final_answer}\n"
        log_lines.append("[forced] ran out of iterations, synthesized final answer")

    log_lines.append("")
    log_lines.append(f"Final Answer: {final_answer}")

    if verbose:
        print("\n".join(log_lines))

    return {
        "transcript": transcript,
        "log": "\n".join(log_lines),
        "final_answer": final_answer,
        "reached_final": reached_final,
        "forced_stop": forced_stop,
        "n_steps": transcript.count("Action:"),
        "n_tool_errors": n_tool_errors,
        "real_observations": real_observations,  # ground truth: only dicts a tool actually returned
    }


def _synthesize_final_answer(real_observations):
    """If the model never produced a clean Final Answer (or produced one we can't trust),
    build one directly from the dicts returned by tool calls that actually ran - never by
    re-parsing transcript text, since a small model will sometimes type text that looks
    like 'Observation: {...}' inside its own response without the tool having run at all."""
    if not real_observations:
        return "No usable tool results were produced."

    lines = ["Summary of tool results:"]
    for r in real_observations:
        if "test_accuracy" in r:
            lines.append(f"- {r.get('dataset_name')}/{r.get('model_type', r.get('optimizer_name', ''))}: test_accuracy={r['test_accuracy']}")
        elif "best_cv_accuracy" in r:
            lines.append(f"- {r.get('dataset_name')}/{r.get('model_type')} tuning: best_cv_accuracy={r['best_cv_accuracy']}, params={r.get('best_params')}")
        elif "reduced_accuracy" in r:
            lines.append(f"- {r.get('dataset_name')} {r.get('method')} (n_components={r.get('n_components')}): accuracy={r['reduced_accuracy']}")
    return "\n".join(lines)


DEMO_TASKS = [
    "Summarize the iris dataset.",
    "Compare decision tree, logistic regression, and random forest on the wine dataset.",
    "Tune hyperparameters for an SVC on the wine dataset with grid search.",
    "Train a deep classifier on wine and show what happens with a batch size of 1.",
    "Train a pytorch model on breast_cancer with a learning rate of 50 and see what happens.",
]


def run_demos(backend, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    for i, task in enumerate(DEMO_TASKS):
        print(f"\n########## DEMO {i + 1}: {task} ##########\n")
        try:
            result = run_agent_loop(task, backend, verbose=True)
            log_text = result["log"]
        except Exception as e:
            # belt-and-suspenders: even if something unexpected slips past
            # run_agent_loop's own error handling, one bad demo should never
            # take the rest of the batch down with it
            log_text = f"=== Agent run: {task} ===\nDEMO CRASHED: {e}\n"
            print(log_text)
        fname = os.path.join(log_dir, f"demo_{i + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(fname, "w") as f:
            f.write(log_text)
        print(f"(saved to {fname})")


def main():
    parser = argparse.ArgumentParser(description="Run the CSE445 ReAct ML agent.")
    parser.add_argument("--task", type=str, default=None, help="A single query for the agent to solve.")
    parser.add_argument("--backend", choices=["ollama", "mock"], default="ollama")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--demo", action="store_true", help="Run the built-in demo task list.")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    args = parser.parse_args()

    backend = OllamaBackend(model=args.model, timeout=args.timeout) if args.backend == "ollama" else MockBackend()

    if args.demo:
        run_demos(backend, log_dir=args.log_dir)
        return

    task = args.task or "Summarize the iris dataset and then train a decision tree on it."
    result = run_agent_loop(task, backend, max_iterations=args.max_iterations, verbose=True)

    os.makedirs(args.log_dir, exist_ok=True)
    fname = os.path.join(args.log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(fname, "w") as f:
        f.write(result["log"])
    print(f"\n(log saved to {fname})")


if __name__ == "__main__":
    main()
