import argparse
import json
import os
import traceback
from datetime import datetime

import pandas as pd

from . import common
from .mmlu_eval import MMLUEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.qwen_sampler import QwenCompletionSampler
from .sampler.responses_sampler import ResponsesSampler

# Import healthbench evaluations with error handling
try:
    from .healthbench_eval import HealthBenchEval
    from .healthbench_meta_eval import HealthBenchMetaEval
    HEALTHBENCH_AVAILABLE = True
except ImportError:
    print("Warning: HealthBench evaluations not available. Only MMLU will be used.")
    HEALTHBENCH_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Select a model by name. Also accepts a comma-separated list of models.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        help="Select an eval by name. Also accepts a comma-separated list of evals.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeats to run. Only supported for certain evals.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=8,
        help="Number of threads to run. Only supported for HealthBench and HealthBenchMeta.",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )

    args = parser.parse_args()

    # Define available models (use sanitized model names for file paths)
    models = {
        # Using a sanitized name for the model to avoid file path issues
        "Qwen3-14B": QwenCompletionSampler(
            model_name="Qwen/Qwen3-14B",
            api_url="http://localhost:8000/v1/completions",
            temperature=0.0,  # Use deterministic outputs for evaluations
        ),
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

    print(f"Running with args {args}")

    # Initialize grading samplers
    grading_sampler = ChatCompletionSampler(
        model="gpt-4.1-2025-04-14",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    )
    equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")

    def get_evals(eval_name, debug_mode):
        """Get evaluation objects with proper error handling"""
        num_examples = args.examples if args.examples is not None else (5 if debug_mode else None)
        
        try:
            match eval_name:
                case "mmlu":
                    return MMLUEval(num_examples=10 if debug_mode else (num_examples or 5000))
                
                case "healthbench" if HEALTHBENCH_AVAILABLE:
                    return HealthBenchEval(
                        grader_model=grading_sampler,
                        num_examples=10 if debug_mode else (num_examples or 5000),
                        n_repeats=args.n_repeats or 1,
                        n_threads=args.n_threads or 1,
                        subset_name=None,
                    )
                case "healthbench_hard" if HEALTHBENCH_AVAILABLE:
                    return HealthBenchEval(
                        grader_model=grading_sampler,
                        num_examples=10 if debug_mode else (num_examples or 5000),
                        n_repeats=args.n_repeats or 1,
                        n_threads=args.n_threads or 1,
                        subset_name="hard",
                    )
                case "healthbench_consensus" if HEALTHBENCH_AVAILABLE:
                    return HealthBenchEval(
                        grader_model=grading_sampler,
                        num_examples=10 if debug_mode else (num_examples or 5000),
                        n_repeats=args.n_repeats or 1,
                        n_threads=args.n_threads or 1,
                        subset_name="consensus",
                    )
                case "healthbench_meta" if HEALTHBENCH_AVAILABLE:
                    return HealthBenchMetaEval(
                        grader_model=grading_sampler,
                        num_examples=10 if debug_mode else (num_examples or 5000),
                        n_repeats=args.n_repeats or 1,
                        n_threads=args.n_threads or 1,
                    )
                case _:
                    raise ValueError(f"Unrecognized eval type: {eval_name}")
        except Exception as e:
            print(f"Error initializing eval '{eval_name}': {e}")
            traceback.print_exc()
            return None

    # Determine which evaluations to run
    if args.eval:
        evals_list = args.eval.split(",")
        evals = {}
        for eval_name in evals_list:
            eval_obj = get_evals(eval_name, args.debug)
            if eval_obj is not None:
                evals[eval_name] = eval_obj
            else:
                print(f"Error: eval '{eval_name}' could not be initialized.")
        
        if not evals:
            print("No valid evaluations found. Exiting.")
            return
    else:
        # Default evaluations
        available_evals = ["mmlu"]
        if HEALTHBENCH_AVAILABLE:
            available_evals.extend([
                "healthbench",
                "healthbench_hard",
                "healthbench_consensus",
                "healthbench_meta",
            ])
        
        evals = {}
        for eval_name in available_evals:
            eval_obj = get_evals(eval_name, args.debug)
            if eval_obj is not None:
                evals[eval_name] = eval_obj

    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    print(f"Running the following evals: {list(evals.keys())}")
    print(f"Running evals for the following models: {list(models.keys())}")

    # Create output directory if it doesn't exist
    output_dir = "/tmp"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run evaluations
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            try:
                print(f"Running {eval_name} evaluation with {model_name}...")
                result = eval_obj(sampler)
                
                # Generate file names with sanitized paths
                file_stem = f"{eval_name}_{model_name}_{date_str}"
                report_filename = os.path.join(output_dir, f"{file_stem}{debug_suffix}.html")
                
                print(f"Writing report to {report_filename}")
                with open(report_filename, "w") as fh:
                    fh.write(common.make_report(result))
                
                assert result.metrics is not None
                metrics = result.metrics | {"score": result.score}
                # Sort metrics by key
                metrics = dict(sorted(metrics.items()))
                print(metrics)
                
                result_filename = os.path.join(output_dir, f"{file_stem}{debug_suffix}.json")
                with open(result_filename, "w") as f:
                    f.write(json.dumps(metrics, indent=2))
                print(f"Writing results to {result_filename}")

                full_result_filename = os.path.join(output_dir, f"{file_stem}{debug_suffix}_allresults.json")
                with open(full_result_filename, "w") as f:
                    result_dict = {
                        "score": result.score,
                        "metrics": result.metrics,
                        "htmls": result.htmls,
                        "convos": result.convos,
                        "metadata": result.metadata,
                    }
                    f.write(json.dumps(result_dict, indent=2))
                    print(f"Writing all results to {full_result_filename}")

                mergekey2resultpath[file_stem] = result_filename
            
            except Exception as e:
                print(f"Error running evaluation {eval_name} with model {model_name}: {e}")
                traceback.print_exc()
    
    # Generate summary table
    if mergekey2resultpath:
        merge_metrics = []
        for eval_model_name, result_filename in mergekey2resultpath.items():
            try:
                with open(result_filename, "r") as f:
                    result = json.load(f)
                result_score = result.get("f1_score", result.get("score", None))
                eval_name = eval_model_name[: eval_model_name.find("_")]
                model_name = eval_model_name[eval_model_name.find("_") + 1 : eval_model_name.rfind("_")]
                merge_metrics.append(
                    {"eval_name": eval_name, "model_name": model_name, "metric": result_score}
                )
            except Exception as e:
                print(f"Error processing results from {result_filename}: {e}")
        
        if merge_metrics:
            merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
                index=["model_name"], columns="eval_name"
            )
            print("\nAll results: ")
            print(merge_metrics_df.to_markdown())
            return merge_metrics
        else:
            print("No results to summarize.")
            return []
    else:
        print("No evaluations were successfully completed.")
        return []


if __name__ == "__main__":
    main()