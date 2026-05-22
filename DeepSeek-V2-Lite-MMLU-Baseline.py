#!/usr/bin/env python3
"""
Evaluate a local DeepSeek-V2-Lite checkpoint on a local MMLU dataset dump.

Expected local files under ./MMLU:
    - dev-00000-of-00001.parquet
    - test-00000-of-00001.parquet
    - optional validation-00000-of-00001.parquet
    - task templates such as _mmlu.yaml / _default_template_yaml.txt

Prompt template follows ./MMLU/_default_template_yaml.txt:
    {question}
    A. {choices[0]}
    B. {choices[1]}
    C. {choices[2]}
    D. {choices[3]}
    Answer:

Few-shot examples are drawn from the local dev split with first_n selection,
which matches the local template file's fewshot_config.sampler = first_n.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig


LETTER_CHOICES = ["A", "B", "C", "D"]
DEFAULT_MMLU_CATEGORY_MAP: Dict[str, str] = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek-V2-Lite on local MMLU parquet files."
    )
    parser.add_argument("--model_path", type=str, default="./DeepSeek-V2-Lite")
    parser.add_argument("--mmlu_dir", type=str, default="./MMLU")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="none",
        choices=["none", "auto"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of evaluation examples scored per batch. Default 1 for maximum scoring stability.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=5,
        help="Number of few-shot demonstrations from the local dev split. Use 0 for zero-shot.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["test", "validation"],
    )
    parser.add_argument("--answer_prefix", type=str, default=" ")
    parser.add_argument("--subjects", type=str, nargs="*", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--skip_generation_test", action="store_true")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(device_name)


def resolve_torch_dtype(dtype_name: str, config: Any, target_device: torch.device) -> torch.dtype:
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

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
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
            model = model_class.from_pretrained(str(model_path), config=config, **load_kwargs)
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
            model = model_class.from_pretrained(str(model_path), config=config, **fallback_kwargs)
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
    except Exception as exc:
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


def generation_sanity_check(model: Any, tokenizer: Any) -> None:
    generate_fn = getattr(model, "generate", None)
    if not callable(generate_fn):
        print(
            "Skipping generation sanity check because this DeepSeek model class "
            "does not expose generate() under the current transformers version."
        )
        return

    prompt = "The largest planet in our solar system is"
    print("Running generation sanity check...")
    encoded = tokenizer(prompt, return_tensors="pt")
    input_device = get_model_input_device(model)
    encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}

    with torch.inference_mode():
        outputs = generate_fn(**encoded, max_new_tokens=12, do_sample=False)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=" * 80)
    print("GENERATION TEST (Sanity Check)")
    print(f"Input : {prompt}")
    print(f"Output: {decoded}")
    print("=" * 80 + "\n")


def load_local_mmlu_dataset(mmlu_dir: Path, eval_split: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required to load local parquet-based MMLU files. "
            "Install it with: pip install datasets pyarrow"
        ) from exc

    data_files: Dict[str, str] = {}
    dev_file = mmlu_dir / "dev-00000-of-00001.parquet"
    if dev_file.exists():
        data_files["dev"] = str(dev_file)

    eval_file = mmlu_dir / f"{eval_split}-00000-of-00001.parquet"
    if eval_file.exists():
        data_files[eval_split] = str(eval_file)
    elif eval_split == "test":
        validation_fallback = mmlu_dir / "validation-00000-of-00001.parquet"
        if validation_fallback.exists():
            print("Requested test split not found; falling back to validation split.")
            eval_split = "validation"
            data_files["validation"] = str(validation_fallback)
        else:
            raise FileNotFoundError(
                f"Could not find {eval_file.name} or validation-00000-of-00001.parquet under {mmlu_dir}."
            )
    else:
        raise FileNotFoundError(f"Could not find {eval_file.name} under {mmlu_dir}.")

    dataset = load_dataset("parquet", data_files=data_files)
    if "dev" not in dataset:
        raise FileNotFoundError(f"Could not find local dev split under {mmlu_dir}.")

    dev_rows = [dict(row) for row in dataset["dev"]]
    eval_rows = [dict(row) for row in dataset[eval_split]]
    return dev_rows, eval_rows


def normalize_choices(raw_choices: Any) -> List[str]:
    if isinstance(raw_choices, (list, tuple)):
        choices = [str(item) for item in raw_choices]
    elif isinstance(raw_choices, dict):
        if "text" in raw_choices:
            return normalize_choices(raw_choices["text"])
        sorted_items = sorted(raw_choices.items())
        choices = [str(value) for _, value in sorted_items]
    else:
        raise TypeError(f"Unsupported choices type: {type(raw_choices).__name__}")

    if len(choices) != 4:
        raise ValueError(f"MMLU examples must have exactly 4 choices, got {len(choices)}.")
    return choices


def normalize_answer(raw_answer: Any) -> int:
    if isinstance(raw_answer, bool):
        return int(raw_answer)
    if isinstance(raw_answer, int):
        if 0 <= raw_answer < 4:
            return raw_answer
    if isinstance(raw_answer, str):
        answer_str = raw_answer.strip()
        if answer_str in LETTER_CHOICES:
            return LETTER_CHOICES.index(answer_str)
        if answer_str.isdigit():
            answer_int = int(answer_str)
            if 0 <= answer_int < 4:
                return answer_int
    raise ValueError(f"Unsupported MMLU answer value: {raw_answer!r}")


def get_subject_name(row: Dict[str, Any]) -> str:
    for key in ("subject", "subtask", "category", "task", "dataset", "topic"):
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return "unknown_subject"


def normalize_mmlu_row(row: Dict[str, Any]) -> Dict[str, Any]:
    question = row.get("question")
    if question is None:
        raise KeyError("MMLU row is missing the 'question' field.")

    if "choices" not in row:
        raise KeyError("MMLU row is missing the 'choices' field.")
    if "answer" not in row:
        raise KeyError("MMLU row is missing the 'answer' field.")

    return {
        "question": str(question).strip(),
        "choices": normalize_choices(row["choices"]),
        "answer": normalize_answer(row["answer"]),
        "subject": get_subject_name(row),
    }


def build_mmlu_question_block(question: str, choices: Sequence[str]) -> str:
    return (
        f"{question.strip()}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer:"
    )


def build_mmlu_prompt(example: Dict[str, Any], fewshot_examples: Sequence[Dict[str, Any]]) -> str:
    sections: List[str] = []
    for shot in fewshot_examples:
        answer_letter = LETTER_CHOICES[int(shot["answer"])]
        sections.append(
            build_mmlu_question_block(shot["question"], shot["choices"]) + f" {answer_letter}"
        )
    sections.append(build_mmlu_question_block(example["question"], example["choices"]))
    return "\n\n".join(sections)


def tokenize_prompt_and_choice(
    tokenizer: Any,
    prompt: str,
    continuation: str,
) -> Tuple[List[int], List[int], List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    continuation_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if len(continuation_ids) == 0:
        raise ValueError(f"Continuation tokenization produced no tokens for {continuation!r}.")
    full_ids = list(prompt_ids) + list(continuation_ids)
    return prompt_ids, continuation_ids, full_ids


def prepare_choice_batch(
    tokenizer: Any,
    examples: Sequence[Dict[str, Any]],
    fewshot_by_subject: Dict[str, List[Dict[str, Any]]],
    num_fewshot: int,
    answer_prefix: str,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    encoded_rows: List[Dict[str, List[int]]] = []
    metadata: List[Dict[str, Any]] = []

    for example_idx, example in enumerate(examples):
        prompt = build_mmlu_prompt(
            example=example,
            fewshot_examples=fewshot_by_subject.get(example["subject"], [])[:num_fewshot],
        )

        for choice_index, choice_letter in enumerate(LETTER_CHOICES):
            continuation = f"{answer_prefix}{choice_letter}"
            prompt_ids, continuation_ids, full_ids = tokenize_prompt_and_choice(
                tokenizer=tokenizer,
                prompt=prompt,
                continuation=continuation,
            )
            encoded_rows.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                }
            )
            metadata.append(
                {
                    "example_index": example_idx,
                    "choice_index": choice_index,
                    "choice_letter": choice_letter,
                    "prompt": prompt,
                    "continuation": continuation,
                    "prompt_token_count": len(prompt_ids),
                    "choice_token_count": len(continuation_ids),
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


def build_fewshot_index(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_subject: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)
    return dict(by_subject)


def filter_subjects(rows: Sequence[Dict[str, Any]], subjects: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    if not subjects:
        return list(rows)
    subject_set = {subject.strip() for subject in subjects if subject.strip() != ""}
    filtered = [row for row in rows if row["subject"] in subject_set]
    missing_subjects = sorted(subject_set - {row["subject"] for row in filtered})
    if missing_subjects:
        print(f"Warning: requested subjects not found in evaluation split: {missing_subjects}")
    return filtered


def evaluate_mmlu(
    model: Any,
    tokenizer: Any,
    eval_rows: Sequence[Dict[str, Any]],
    fewshot_by_subject: Dict[str, List[Dict[str, Any]]],
    batch_size: int,
    num_fewshot: int,
    answer_prefix: str,
    save_path: Optional[Path],
    model_path: Path,
    mmlu_dir: Path,
) -> Dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(eval_rows) == 0:
        raise ValueError("MMLU evaluation split is empty after filtering.")

    total_samples = len(eval_rows)
    acc_correct = 0
    acc_norm_correct = 0
    predictions: List[Dict[str, Any]] = []
    per_subject: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "acc": 0, "acc_norm": 0})
    per_category: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "acc": 0, "acc_norm": 0})

    print(f"Evaluating on {total_samples} MMLU examples...\n")
    for start in tqdm(range(0, total_samples, batch_size), desc="Evaluating MMLU"):
        examples = eval_rows[start : start + batch_size]
        batch, metadata = prepare_choice_batch(
            tokenizer=tokenizer,
            examples=examples,
            fewshot_by_subject=fewshot_by_subject,
            num_fewshot=num_fewshot,
            answer_prefix=answer_prefix,
        )
        scores = score_choice_batch(model, batch, metadata)

        grouped_scores: Dict[int, Dict[int, Dict[str, float]]] = {}
        grouped_meta: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for item, score in zip(metadata, scores):
            grouped_scores.setdefault(item["example_index"], {})[item["choice_index"]] = score
            grouped_meta.setdefault(item["example_index"], {})[item["choice_index"]] = item

        for local_index, example in enumerate(examples):
            choice_scores = grouped_scores[local_index]
            pred_acc = max(choice_scores.items(), key=lambda pair: pair[1]["sum_logprob"])[0]
            pred_acc_norm = max(choice_scores.items(), key=lambda pair: pair[1]["mean_logprob"])[0]

            is_acc_correct = pred_acc == int(example["answer"])
            is_acc_norm_correct = pred_acc_norm == int(example["answer"])
            subject = example["subject"]
            category = DEFAULT_MMLU_CATEGORY_MAP.get(subject, "unknown")

            per_subject[subject]["total"] += 1
            per_category[category]["total"] += 1
            if is_acc_correct:
                acc_correct += 1
                per_subject[subject]["acc"] += 1
                per_category[category]["acc"] += 1
            if is_acc_norm_correct:
                acc_norm_correct += 1
                per_subject[subject]["acc_norm"] += 1
                per_category[category]["acc_norm"] += 1

            prediction_record: Dict[str, Any] = {
                "index": start + local_index,
                "subject": subject,
                "category": category,
                "question": example["question"],
                "choices": example["choices"],
                "label": int(example["answer"]),
                "label_letter": LETTER_CHOICES[int(example["answer"])],
                "prediction_acc": int(pred_acc),
                "prediction_acc_letter": LETTER_CHOICES[int(pred_acc)],
                "prediction_acc_norm": int(pred_acc_norm),
                "prediction_acc_norm_letter": LETTER_CHOICES[int(pred_acc_norm)],
                "correct_acc": bool(is_acc_correct),
                "correct_acc_norm": bool(is_acc_norm_correct),
                "prompt": grouped_meta[local_index][0]["prompt"],
            }
            for choice_index, choice_letter in enumerate(LETTER_CHOICES):
                prediction_record[f"sum_logprob_{choice_letter}"] = choice_scores[choice_index]["sum_logprob"]
                prediction_record[f"mean_logprob_{choice_letter}"] = choice_scores[choice_index]["mean_logprob"]
            predictions.append(prediction_record)

    acc = acc_correct / float(total_samples)
    acc_norm = acc_norm_correct / float(total_samples)

    subject_metrics = {}
    for subject in sorted(per_subject):
        total = per_subject[subject]["total"]
        subject_metrics[subject] = {
            "total": total,
            "acc": per_subject[subject]["acc"] / float(total),
            "acc_norm": per_subject[subject]["acc_norm"] / float(total),
        }

    category_metrics = {}
    for category in sorted(per_category):
        total = per_category[category]["total"]
        category_metrics[category] = {
            "total": total,
            "acc": per_category[category]["acc"] / float(total),
            "acc_norm": per_category[category]["acc_norm"] / float(total),
        }

    summary = {
        "model_path": str(model_path),
        "mmlu_dir": str(mmlu_dir),
        "num_samples": total_samples,
        "num_fewshot": num_fewshot,
        "answer_prefix": answer_prefix,
        "metrics": {
            "acc": acc,
            "acc_norm": acc_norm,
        },
        "correct": {
            "acc": acc_correct,
            "acc_norm": acc_norm_correct,
        },
        "category_metrics": category_metrics,
        "subject_metrics": subject_metrics,
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

    print("\nCategory breakdown:")
    for category in sorted(category_metrics):
        item = category_metrics[category]
        print(
            f"  {category:16s} acc={item['acc']:.4f} "
            f"acc_norm={item['acc_norm']:.4f} total={item['total']}"
        )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as fout:
            json.dump(summary, fout, indent=2, ensure_ascii=False)
        print(f"\nSaved detailed results to: {save_path}")

    return summary


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    mmlu_dir = Path(args.mmlu_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not mmlu_dir.exists():
        raise FileNotFoundError(f"MMLU directory does not exist: {mmlu_dir}")

    device = resolve_device(args.device)

    print("\n" + "=" * 80)
    print("DeepSeek-V2-Lite MMLU Baseline Evaluation")
    print("=" * 80)
    print(f"Model        : {model_path}")
    print(f"MMLU dir     : {mmlu_dir}")
    print(f"Device       : {device}")
    print(f"DType        : {args.dtype}")
    print(f"Device map   : {args.device_map}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Few-shot     : {args.num_fewshot}")
    print(f"Eval split   : {args.eval_split}")
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

    dev_rows_raw, eval_rows_raw = load_local_mmlu_dataset(mmlu_dir, args.eval_split)
    dev_rows = [normalize_mmlu_row(row) for row in dev_rows_raw]
    eval_rows = [normalize_mmlu_row(row) for row in eval_rows_raw]

    fewshot_by_subject = build_fewshot_index(dev_rows)
    eval_rows = filter_subjects(eval_rows, args.subjects)
    if args.num_samples is not None:
        eval_rows = eval_rows[: args.num_samples]

    evaluate_mmlu(
        model=model,
        tokenizer=tokenizer,
        eval_rows=eval_rows,
        fewshot_by_subject=fewshot_by_subject,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        answer_prefix=args.answer_prefix,
        save_path=Path(args.save_path) if args.save_path is not None else None,
        model_path=model_path,
        mmlu_dir=mmlu_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.", file=sys.stderr)
        raise
