#!/usr/bin/env python3
"""
Evaluate a local DeepSeek-V2-Lite checkpoint on the local PIQA validation split.

This script mirrors the PIQA task definition in ./PIQA/piqa.yaml:
    doc_to_text   -> "Question: {goal}\nAnswer:"
    doc_to_choice -> [sol1, sol2]
    doc_to_target -> label

It computes two standard multiple-choice metrics:
    1. acc:      choose the option with the higher total log-likelihood
    2. acc_norm: choose the option with the higher average token log-likelihood

By default, continuations are prefixed with a single space, which matches the
common lm-evaluation-harness target delimiter behavior. If you want literal
string concatenation, pass --answer_prefix "".
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig


ROOT_DIR = Path(__file__).resolve().parent
TOKENIZATION_BOUNDARY_WARNING_EMITTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek-V2-Lite on PIQA with local files."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./DeepSeek-V2-Lite",
        help="Path to the local DeepSeek-V2-Lite checkpoint.",
    )
    parser.add_argument(
        "--piqa_dir",
        type=str,
        default="./PIQA",
        help="Directory containing dev.jsonl and dev-labels.lst.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for evaluation. Use auto/cpu/cuda/cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Runtime dtype used while loading the model.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="none",
        choices=["none", "auto"],
        help="Optional device_map passed to from_pretrained. Use auto if direct GPU loading is needed.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of PIQA examples scored per batch.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Evaluate only the first N validation examples.",
    )
    parser.add_argument(
        "--answer_prefix",
        type=str,
        default=" ",
        help="Prefix inserted before each answer choice. Use an empty string to disable it.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional path to save a JSON summary with per-example predictions.",
    )
    parser.add_argument(
        "--skip_generation_test",
        action="store_true",
        help="Skip the short generation sanity check before PIQA evaluation.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(device_name)


def resolve_torch_dtype(
    dtype_name: str,
    config: Any,
    target_device: torch.device,
) -> torch.dtype:
    if dtype_name == "auto":
        config_dtype = getattr(config, "torch_dtype", None)
        if isinstance(config_dtype, str):
            config_dtype = getattr(torch, config_dtype, None)
        if isinstance(config_dtype, torch.dtype):
            if target_device.type == "cpu" and config_dtype in {torch.float16, torch.bfloat16}:
                print("Auto dtype resolved to float32 because the evaluation device is CPU.")
                return torch.float32
            return config_dtype
        return torch.float32 if target_device.type == "cpu" else torch.bfloat16

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    runtime_dtype = mapping[dtype_name]
    if target_device.type == "cpu" and runtime_dtype in {torch.float16, torch.bfloat16}:
        print(f"Requested dtype {dtype_name} on CPU; using float32 instead.")
        return torch.float32
    return runtime_dtype


def get_model_input_device(model: Any) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
    return next(model.parameters()).device


def patch_transformers_compatibility() -> None:
    """
    DeepSeek-V2-Lite ships custom modeling code that imports a few helpers from
    older/newer Transformers internals. Add small shims before dynamic module
    loading so local `trust_remote_code=True` imports survive API drift.
    """
    try:
        import transformers.utils.import_utils as import_utils
    except Exception:
        return

    if not hasattr(import_utils, "is_torch_fx_available"):
        def is_torch_fx_available() -> bool:
            return hasattr(torch, "fx")

        import_utils.is_torch_fx_available = is_torch_fx_available


def load_module_from_file(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create a module spec for {file_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_deepseek_rope_scaling(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    rope_scaling = config_dict.get("rope_scaling")
    if not isinstance(rope_scaling, dict):
        return config_dict

    normalized = dict(rope_scaling)
    for key in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
        value = normalized.get(key)
        if isinstance(value, int):
            normalized[key] = float(value)

    config_dict = dict(config_dict)
    config_dict["rope_scaling"] = normalized
    return config_dict


def load_local_deepseek_classes(model_path: Path) -> Optional[Tuple[Any, Any]]:
    config_file = model_path / "configuration_deepseek.py"
    model_file = model_path / "modeling_deepseek.py"
    if not config_file.exists() or not model_file.exists():
        return None

    package_hash = hashlib.md5(str(model_path.resolve()).encode("utf-8")).hexdigest()[:12]
    package_name = f"_local_deepseek_{package_hash}"

    package_module = sys.modules.get(package_name)
    if package_module is None:
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(model_path)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package_module

    config_module_name = f"{package_name}.configuration_deepseek"
    model_module_name = f"{package_name}.modeling_deepseek"

    if config_module_name in sys.modules:
        config_module = sys.modules[config_module_name]
    else:
        config_module = load_module_from_file(config_module_name, config_file)

    if model_module_name in sys.modules:
        model_module = sys.modules[model_module_name]
    else:
        model_module = load_module_from_file(model_module_name, model_file)

    config_class = getattr(config_module, "DeepseekV2Config", None)
    model_class = getattr(model_module, "DeepseekV2ForCausalLM", None)
    if config_class is None or model_class is None:
        raise AttributeError(
            "Local DeepSeek modules were loaded, but DeepseekV2Config or "
            "DeepseekV2ForCausalLM was not found."
        )

    return config_class, model_class


def load_model_and_tokenizer(
    model_path: Path,
    device: torch.device,
    dtype_name: str,
    device_map: str,
) -> Tuple[Any, Any, Any]:
    print(f"Loading model from: {model_path}")
    patch_transformers_compatibility()
    local_classes = load_local_deepseek_classes(model_path)
    if local_classes is not None:
        config_class, model_class = local_classes
        with (model_path / "config.json").open("r", encoding="utf-8") as fin:
            config_dict = json.load(fin)
        config_dict = normalize_deepseek_rope_scaling(config_dict)
        config = config_class.from_dict(config_dict)
        config._name_or_path = str(model_path)
    else:
        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        model_class = None
    runtime_dtype = resolve_torch_dtype(dtype_name, config, device)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": runtime_dtype,
        "low_cpu_mem_usage": True,
    }
    if device_map == "auto":
        load_kwargs["device_map"] = "auto"

    try:
        if model_class is not None:
            model = model_class.from_pretrained(
                str(model_path),
                config=config,
                **load_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    except Exception as exc:
        exc_message = str(exc).lower()
        can_retry_without_low_cpu_mem = (
            device_map == "none"
            and load_kwargs.get("low_cpu_mem_usage", False)
            and ("accelerate" in exc_message or "low_cpu_mem_usage" in exc_message)
        )
        if not can_retry_without_low_cpu_mem:
            raise

        print(
            "Warning: low_cpu_mem_usage loading failed, likely because accelerate is unavailable. "
            "Retrying with the standard from_pretrained path."
        )
        fallback_kwargs = dict(load_kwargs)
        fallback_kwargs.pop("low_cpu_mem_usage", None)
        if model_class is not None:
            model = model_class.from_pretrained(
                str(model_path),
                config=config,
                **fallback_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(str(model_path), **fallback_kwargs)
    if device_map == "none":
        print(f"Moving model to device: {device}")
        model = model.to(device)
        print("Model move completed.")
    model.eval()
    print("Model eval mode enabled.")

    try:
        generation_config = GenerationConfig.from_pretrained(str(model_path))
        generation_config.pad_token_id = generation_config.eos_token_id
        model.generation_config = generation_config
        print("Generation config loaded.")
    except Exception as exc:  # pragma: no cover - best effort only
        print(f"Warning: failed to load generation config: {exc}")

    model_input_device = get_model_input_device(model)
    print("Model load summary:")
    print(f"  model_type     : {getattr(config, 'model_type', 'unknown')}")
    print(f"  architectures  : {getattr(config, 'architectures', None)}")
    print(f"  target_device  : {device}")
    print(f"  input_device   : {model_input_device}")
    print(f"  device_map     : {device_map}")
    print(f"  runtime_dtype  : {next(model.parameters()).dtype}")
    print("")

    return model, tokenizer, config


def load_piqa_local(piqa_dir: Path) -> List[Dict[str, Any]]:
    data_file = piqa_dir / "dev.jsonl"
    labels_file = piqa_dir / "dev-labels.lst"
    if not data_file.exists() or not labels_file.exists():
        raise FileNotFoundError(
            f"Local PIQA files not found under {piqa_dir}. "
            "Expected dev.jsonl and dev-labels.lst."
        )

    examples: List[Dict[str, Any]] = []
    with data_file.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    with labels_file.open("r", encoding="utf-8") as fin:
        labels = [int(line.strip()) for line in fin if line.strip() != ""]

    if len(examples) != len(labels):
        raise ValueError(
            f"PIQA local files have mismatched lengths: {len(examples)} examples vs {len(labels)} labels."
        )

    dataset: List[Dict[str, Any]] = []
    for example, label in zip(examples, labels):
        dataset.append(
            {
                "goal": example["goal"],
                "sol1": example["sol1"],
                "sol2": example["sol2"],
                "label": label,
            }
        )
    return dataset


def build_piqa_prompt(goal: str) -> str:
    return f"Question: {goal}\nAnswer:"


def resolve_prefix_length(prompt_ids: Sequence[int], full_ids: Sequence[int]) -> int:
    global TOKENIZATION_BOUNDARY_WARNING_EMITTED

    prompt_len = len(prompt_ids)
    if full_ids[:prompt_len] == list(prompt_ids):
        return prompt_len

    prefix_len = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids):
        if prompt_token != full_token:
            break
        prefix_len += 1

    if prefix_len == 0:
        raise ValueError("Could not align prompt tokens with full input tokens.")

    if not TOKENIZATION_BOUNDARY_WARNING_EMITTED:
        warnings.warn(
            "Prompt tokens were not a strict prefix of prompt+answer tokens. "
            "Scoring will use the longest common prefix boundary. "
            "If you want stronger boundary stability, keep --answer_prefix as a single space.",
            stacklevel=2,
        )
        TOKENIZATION_BOUNDARY_WARNING_EMITTED = True

    return prefix_len


def prepare_choice_batch(
    tokenizer: Any,
    examples: Sequence[Dict[str, Any]],
    answer_prefix: str,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    encoded_rows: List[Dict[str, List[int]]] = []
    metadata: List[Dict[str, Any]] = []

    for example_idx, example in enumerate(examples):
        prompt = build_piqa_prompt(example["goal"])
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]

        for choice_idx, choice_key in enumerate(("sol1", "sol2")):
            continuation = f"{answer_prefix}{example[choice_key]}"
            full_encoding = tokenizer(prompt + continuation, add_special_tokens=True)
            full_ids = full_encoding["input_ids"]
            prefix_len = resolve_prefix_length(prompt_ids, full_ids)
            continuation_len = len(full_ids) - prefix_len
            if continuation_len <= 0:
                raise ValueError(
                    f"Choice {choice_key} for example {example_idx} produced no continuation tokens."
                )

            encoded_rows.append(
                {
                    "input_ids": full_encoding["input_ids"],
                    "attention_mask": full_encoding["attention_mask"],
                }
            )
            metadata.append(
                {
                    "example_index": example_idx,
                    "choice_index": choice_idx,
                    "choice_key": choice_key,
                    "prompt": prompt,
                    "continuation": continuation,
                    "prompt_token_count": prefix_len,
                    "choice_token_count": continuation_len,
                }
            )

    batch = tokenizer.pad(encoded_rows, padding=True, return_tensors="pt")
    return batch, metadata


def score_choice_batch(
    model: Any,
    batch: Dict[str, torch.Tensor],
    metadata: Sequence[Dict[str, Any]],
) -> List[Dict[str, float]]:
    input_device = get_model_input_device(model)
    batch = {name: tensor.to(input_device) for name, tensor in batch.items()}

    with torch.inference_mode():
        outputs = model(**batch, use_cache=False)
        shift_logits = outputs.logits[:, :-1, :].float()
        logits_device = shift_logits.device
        shift_labels = batch["input_ids"][:, 1:].to(logits_device)
        shift_mask = batch["attention_mask"][:, 1:].to(logits_device).bool()

        token_log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = token_log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        token_log_probs = token_log_probs.masked_fill(~shift_mask, 0.0)

    scores: List[Dict[str, float]] = []
    for row_idx, item in enumerate(metadata):
        start_idx = item["prompt_token_count"] - 1
        end_idx = start_idx + item["choice_token_count"]
        row_log_probs = token_log_probs[row_idx, start_idx:end_idx]
        row_mask = shift_mask[row_idx, start_idx:end_idx]

        valid_tokens = int(row_mask.sum().item())
        if valid_tokens != item["choice_token_count"]:
            raise ValueError(
                "Continuation token mask does not match the expected continuation length. "
                f"Expected {item['choice_token_count']} tokens, got {valid_tokens}."
            )

        total_log_prob = float(row_log_probs.sum().item())
        avg_log_prob = total_log_prob / float(valid_tokens)
        scores.append(
            {
                "sum_logprob": total_log_prob,
                "mean_logprob": avg_log_prob,
            }
        )
    return scores


def generation_sanity_check(model: Any, tokenizer: Any) -> None:
    generate_fn = getattr(model, "generate", None)
    if not callable(generate_fn):
        print(
            "Skipping generation sanity check because this DeepSeek model class "
            "does not expose generate() under the current transformers version."
        )
        return

    prompt = "The capital of France is"
    print("Running generation sanity check...")
    encoded = tokenizer(prompt, return_tensors="pt")
    input_device = get_model_input_device(model)
    encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}

    with torch.inference_mode():
        outputs = generate_fn(
            **encoded,
            max_new_tokens=12,
            do_sample=False,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=" * 80)
    print("GENERATION TEST (Sanity Check)")
    print(f"Input : {prompt}")
    print(f"Output: {decoded}")
    print("=" * 80 + "\n")


def evaluate_piqa(
    model: Any,
    tokenizer: Any,
    dataset: Sequence[Dict[str, Any]],
    batch_size: int,
    answer_prefix: str,
    save_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    piqa_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(dataset) == 0:
        raise ValueError("PIQA dataset is empty.")

    total_samples = len(dataset)
    acc_correct = 0
    acc_norm_correct = 0
    predictions: List[Dict[str, Any]] = []

    print(f"Evaluating on {total_samples} PIQA validation examples...\n")
    for start in tqdm(range(0, total_samples, batch_size), desc="Evaluating PIQA"):
        examples = dataset[start : start + batch_size]
        batch, metadata = prepare_choice_batch(tokenizer, examples, answer_prefix)
        scores = score_choice_batch(model, batch, metadata)

        grouped_scores: Dict[int, Dict[int, Dict[str, float]]] = {}
        grouped_meta: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for item, score in zip(metadata, scores):
            example_index = item["example_index"]
            choice_index = item["choice_index"]
            grouped_scores.setdefault(example_index, {})[choice_index] = score
            grouped_meta.setdefault(example_index, {})[choice_index] = item

        for local_index, example in enumerate(examples):
            choice0 = grouped_scores[local_index][0]
            choice1 = grouped_scores[local_index][1]
            choice0_meta = grouped_meta[local_index][0]
            choice1_meta = grouped_meta[local_index][1]

            pred_acc = 0 if choice0["sum_logprob"] >= choice1["sum_logprob"] else 1
            pred_acc_norm = 0 if choice0["mean_logprob"] >= choice1["mean_logprob"] else 1

            if pred_acc == example["label"]:
                acc_correct += 1
            if pred_acc_norm == example["label"]:
                acc_norm_correct += 1

            predictions.append(
                {
                    "index": start + local_index,
                    "goal": example["goal"],
                    "sol1": example["sol1"],
                    "sol2": example["sol2"],
                    "label": int(example["label"]),
                    "prediction_acc": int(pred_acc),
                    "prediction_acc_norm": int(pred_acc_norm),
                    "correct_acc": bool(pred_acc == example["label"]),
                    "correct_acc_norm": bool(pred_acc_norm == example["label"]),
                    "sum_logprob_sol1": choice0["sum_logprob"],
                    "sum_logprob_sol2": choice1["sum_logprob"],
                    "mean_logprob_sol1": choice0["mean_logprob"],
                    "mean_logprob_sol2": choice1["mean_logprob"],
                    "prompt": choice0_meta["prompt"],
                    "continuation_sol1": choice0_meta["continuation"],
                    "continuation_sol2": choice1_meta["continuation"],
                    "choice_token_count_sol1": choice0_meta["choice_token_count"],
                    "choice_token_count_sol2": choice1_meta["choice_token_count"],
                }
            )

    acc = acc_correct / float(total_samples)
    acc_norm = acc_norm_correct / float(total_samples)

    summary = {
        "model_path": str(model_path) if model_path is not None else None,
        "piqa_dir": str(piqa_dir) if piqa_dir is not None else None,
        "num_samples": total_samples,
        "answer_prefix": answer_prefix,
        "metrics": {
            "acc": acc,
            "acc_norm": acc_norm,
        },
        "correct": {
            "acc": acc_correct,
            "acc_norm": acc_norm_correct,
        },
        "predictions": predictions,
    }

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print(f"  acc      : {acc:.4f} ({acc * 100:.2f}%)  [{acc_correct}/{total_samples}]")
    print(
        f"  acc_norm : {acc_norm:.4f} ({acc_norm * 100:.2f}%)  "
        f"[{acc_norm_correct}/{total_samples}]"
    )
    print("=" * 80)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as fout:
            json.dump(summary, fout, indent=2, ensure_ascii=False)
        print(f"Saved detailed results to: {save_path}")

    return summary


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    piqa_dir = Path(args.piqa_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not piqa_dir.exists():
        raise FileNotFoundError(f"PIQA directory does not exist: {piqa_dir}")

    device = resolve_device(args.device)

    print("\n" + "=" * 80)
    print("DeepSeek-V2-Lite PIQA Baseline Evaluation")
    print("=" * 80)
    print(f"Model        : {model_path}")
    print(f"PIQA dir     : {piqa_dir}")
    print(f"Device       : {device}")
    print(f"DType        : {args.dtype}")
    print(f"Device map   : {args.device_map}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Answer prefix: {args.answer_prefix!r}")
    print("=" * 80 + "\n")

    model, tokenizer, _ = load_model_and_tokenizer(
        model_path=model_path,
        device=device,
        dtype_name=args.dtype,
        device_map=args.device_map,
    )

    if not args.skip_generation_test:
        generation_sanity_check(model, tokenizer)

    dataset = load_piqa_local(piqa_dir)
    if args.num_samples is not None:
        dataset = dataset[: args.num_samples]

    evaluate_piqa(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        answer_prefix=args.answer_prefix,
        save_path=Path(args.save_path) if args.save_path is not None else None,
        model_path=model_path,
        piqa_dir=piqa_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.", file=sys.stderr)
        raise
