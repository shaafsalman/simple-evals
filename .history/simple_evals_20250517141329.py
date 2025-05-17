import argparse
import json
import pandas as pd
from datetime import datetime

from . import common
from .healthbench_eval import HealthBenchEval
from .healthbench_meta_eval import HealthBenchMetaEval
from .sampler.qwen_sampler import QwenCompletionSampler  

def main():
    parser = argparse.ArgumentParser(description="Run medical evals with Qwen")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, help="Select a model by name")
    parser.add_argument("--eval", type=str, help="Select an eval by name")
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--examples", type=int, help="Number of examples to use")

    args = parser.parse_args()

    models = {
        "qwen": QwenCompletionSampler(model_name="qwen") 
    }

    if args.list_models:
        print("Available models:")
        for model_name in models.keys():
            print(f" - {model_name}")
        return

    if args.model:
        models_chosen = args.model.split(",")
        for model_name in models_chosen:
            if model_name not in models:
                print(f"Error: Model '{model_name}' not found.")
                return
        models = {model_name: models[model_name] for model_name in models_chosen}

    def get_evals(eval_name, debug_mode):
        num_examples = args.examples if args.examples is not None else (5 if debug_mode else None)
        match eval_name:
            case "healthbench":
                return HealthBenchEval(
                    grader_model=models["qwen"],
                    num_examples=num_examples,
                    n_repeats=args.n_repeats,
                    n_threads=args.n_threads,
                )
            case "healthbench_hard":
                return HealthBenchEval(
                    grader_model=models["qwen"],
                    num_examples=num_examples,
                    n_repeats=args.n_repeats,
                    n_threads=args.n_threads,
                    subset_name="hard",
                )
            case "healthbench_consensus":
                return HealthBenchEval(
                    grader_model=models["qwen"],
                    num_examples=num_examples,
                    n_repeats=args.n_repeats,
                    n_threads=args.n_threads,
                    subset_name="consensus",
                )
            case "healthbench_meta":
                return HealthBenchMetaEval(
                    grader_model=models["qwen"],
                    num_examples=num_examples,
                    n_repeats=args.n_repeats,
                    n_threads=args.n_threads,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    if args.eval:
        evals_list = args.eval.split(",")
    else:
        evals_list = ["healthbench", "healthbench_hard", "healthbench_consensus", "healthbench_meta"]

    evals = {}
    for eval_name in evals_list:
        try:
            evals[eval_name] = get_evals(eval_name, args.debug)
        except Exception:
            print(f"Error: eval '{eval_name}' not found.")
            return

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    debug_suffix = "_DEBUG" if args.debug else ""
    mergekey2resultpath = {}

    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            file_stem = f"{eval_name}_{model_name}_{date_str}"
            report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            metrics = dict(sorted(metrics.items()))
            result_filename = f"/tmp/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            full_result_filename = f"/tmp/{file_stem}{debug_suffix}_allresults.json"
            with open(full_result_filename, "w") as f:
                f.write(json.dumps({
                    "score": result.score,
                    "metrics": result.metrics,
                    "htmls": result.htmls,
                    "convos": result.convos,
                    "metadata": result.metadata,
                }, indent=2))
            print(f"Writing all results to {full_result_filename}")
            mergekey2resultpath[file_stem] = result_filename

    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append({
            "eval_name": eval_name,
            "model_name": model_name,
            "metric": result,
        })

    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(index=["model_name"], columns="eval_name")
    print("\nAll results:")
    print(merge_metrics_df.to_markdown())
    return merge_metrics

if __name__ == "__main__":
    main()
