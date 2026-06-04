import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from quantization.hub import LittleBitModel
from utils.datautils import load_tokenizer
from utils.sprefill_data import (
    DEFAULT_SFT_SOURCES,
    prepare_sprefill_sft_dataset,
    render_prompt,
    split_prompt_answer,
)
from utils.sprefill_losses import (
    aggregate_attention_block_scores,
    answer_nll,
    compress_prompt_by_blocks,
    cuda_time_call,
    get_model_device,
    select_blocks_from_scores,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected: {value}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Speculative Prefill LittleBit drafter checkpoint."
    )
    parser.add_argument("--drafter_model_id", type=str, required=True)
    parser.add_argument("--target_model_id", type=str, required=True)
    parser.add_argument("--tokenizer_id", type=str, default=None)
    parser.add_argument("--target_is_littlebit", type=str2bool, default=False)
    parser.add_argument("--drafter_fp_eval", type=str2bool, default=False)
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--drafter_attn_implementation", type=str, default="eager")

    parser.add_argument("--sft_sources", nargs="+", default=DEFAULT_SFT_SOURCES)
    parser.add_argument("--prepared_sft_path", type=str, default=None)
    parser.add_argument("--sft_cache_root", type=str, default="./data")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--dataset_prepared", type=str2bool, default=True)
    parser.add_argument("--max_sft_samples", type=int, default=512)
    parser.add_argument("--max_prompt_tokens", type=int, default=4096)
    parser.add_argument("--max_answer_tokens", type=int, default=512)
    parser.add_argument("--min_prompt_tokens", type=int, default=32)
    parser.add_argument("--min_answer_tokens", type=int, default=4)
    parser.add_argument("--add_eos_to_answer", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--keep_ratio", type=float, default=0.05)
    parser.add_argument("--sink_tokens", type=int, default=256)
    parser.add_argument("--window_tokens", type=int, default=512)
    parser.add_argument("--score_query_tokens", type=int, default=128)
    parser.add_argument("--score_layer_start", type=int, default=None)
    parser.add_argument("--score_layer_end", type=int, default=None)
    parser.add_argument("--score_aggregation", type=str, default="max_mean", choices=["max_mean", "mean", "last_mean"])

    parser.add_argument("--ttft_repeats", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--retention_nll_delta_threshold", type=float, default=0.05)
    parser.add_argument("--output_jsonl", type=str, default="./sprefill_eval_records.jsonl")
    parser.add_argument("--output_summary", type=str, default="./sprefill_eval_summary.json")

    parser.add_argument("--benchmark_jsonl", type=str, default=None)
    parser.add_argument("--prompt_field", type=str, default="prompt")
    parser.add_argument("--answer_field", type=str, default="answer")
    parser.add_argument("--eval_generation", type=str2bool, default=False)
    parser.add_argument("--generation_max_new_tokens", type=int, default=128)
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    if name == "fp16":
        return torch.float16
    return torch.float32


def set_attention_implementation(model, implementation: str) -> None:
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation(implementation)
            return
        except Exception:
            pass
    if hasattr(model, "config"):
        model.config._attn_implementation = implementation


def load_drafter(args, dtype):
    if args.drafter_fp_eval:
        model = AutoModelForCausalLM.from_pretrained(
            args.drafter_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation=args.drafter_attn_implementation,
            device_map=args.device_map if args.device_map.lower() != "none" else None,
        )
    else:
        model = LittleBitModel.from_pretrained(
            args.drafter_model_id,
            torch_dtype=dtype,
            device=args.device_map,
        )
    set_attention_implementation(model, args.drafter_attn_implementation)
    model.eval()
    model.config.use_cache = False
    model.config.output_attentions = True
    return model


def load_target(args, dtype):
    if args.target_is_littlebit:
        model = LittleBitModel.from_pretrained(args.target_model_id, torch_dtype=dtype, device=args.device_map)
    else:
        kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if args.device_map.lower() != "none":
            kwargs["device_map"] = args.device_map
        model = AutoModelForCausalLM.from_pretrained(args.target_model_id, **kwargs)
        if "device_map" not in kwargs and torch.cuda.is_available():
            model.to("cuda")
    model.eval()
    model.config.use_cache = True
    return model


def _make_answer_batch(prompt_ids: torch.Tensor, answer_ids: torch.Tensor, device: torch.device):
    input_ids = torch.cat([prompt_ids, answer_ids], dim=0).unsqueeze(0).to(device)
    labels = torch.cat([torch.full_like(prompt_ids, -100), answer_ids], dim=0).unsqueeze(0).to(device)
    attention = torch.ones_like(input_ids, device=device)
    return input_ids, attention, labels


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(_normalize_text(prediction) == _normalize_text(reference))


def contains_answer(prediction: str, reference: str) -> float:
    pred = _normalize_text(prediction)
    ref = _normalize_text(reference)
    return float(bool(ref) and ref in pred)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), ref_tokens.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def tokenize_plain_prompt_answer(prompt_text: str, answer_text: str, tokenizer, args) -> Dict:
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_prompt_tokens,
    )["input_ids"]
    answer_ids = tokenizer(
        answer_text,
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_answer_tokens,
    )["input_ids"]
    if args.add_eos_to_answer and tokenizer.eos_token_id is not None:
        if not answer_ids or answer_ids[-1] != tokenizer.eos_token_id:
            answer_ids.append(tokenizer.eos_token_id)
    return {
        "prompt_input_ids": prompt_ids,
        "answer_input_ids": answer_ids,
        "answer_text": answer_text,
    }


def load_benchmark_jsonl(args, tokenizer) -> List[Dict]:
    rows = []
    with open(args.benchmark_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            raw = json.loads(line)
            if "prompt_input_ids" in raw and "answer_input_ids" in raw:
                row = {
                    "prompt_input_ids": raw["prompt_input_ids"],
                    "answer_input_ids": raw["answer_input_ids"],
                    "answer_text": raw.get("answer_text", raw.get(args.answer_field, "")),
                }
            else:
                prompt_text = raw.get(args.prompt_field)
                answer_text = raw.get(args.answer_field)
                if prompt_text is None or answer_text is None:
                    split = split_prompt_answer(raw)
                    if not split["valid"]:
                        continue
                    prompt_text = render_prompt(tokenizer, split["prompt_messages"])
                    answer_text = split["answer"]
                row = tokenize_plain_prompt_answer(str(prompt_text), str(answer_text), tokenizer, args)
            row["benchmark_id"] = raw.get("id", raw.get("index", line_idx))
            rows.append(row)
    return rows


def load_eval_rows(args, tokenizer) -> Iterable[Dict]:
    if args.benchmark_jsonl:
        return load_benchmark_jsonl(args, tokenizer)
    return prepare_sprefill_sft_dataset(args, tokenizer)


@torch.no_grad()
def score_prompt_blocks(drafter, prompt_ids: torch.Tensor, args) -> torch.Tensor:
    device = get_model_device(drafter)
    prompt = prompt_ids.unsqueeze(0).to(device)
    attention = torch.ones_like(prompt, device=device)
    outputs = drafter(
        input_ids=prompt,
        attention_mask=attention,
        use_cache=False,
        output_attentions=True,
    )
    scores, _ = aggregate_attention_block_scores(
        outputs.attentions,
        prompt_lens=torch.tensor([prompt_ids.numel()], device=device),
        block_size=args.block_size,
        score_query_tokens=args.score_query_tokens,
        layer_start=args.score_layer_start,
        layer_end=args.score_layer_end,
        aggregation=args.score_aggregation,
    )
    return scores[0].detach().cpu()


@torch.no_grad()
def measure_ttft(model, tokenizer, prompt_ids: torch.Tensor, args) -> float:
    device = get_model_device(model)
    input_ids = prompt_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    times = []
    for _ in range(max(1, args.ttft_repeats)):
        _, elapsed = cuda_time_call(
            lambda: model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_token_id,
            )
        )
        times.append(elapsed)
    return sum(times) / len(times)


@torch.no_grad()
def generate_answer(model, tokenizer, prompt_ids: torch.Tensor, args) -> str:
    device = get_model_device(model)
    input_ids = prompt_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.generation_max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_token_id,
    )
    new_tokens = output[0, input_ids.size(1):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.no_grad()
def evaluate_one(idx: int, row: Dict, drafter, target, tokenizer, args) -> Dict:
    prompt_ids = torch.tensor(row["prompt_input_ids"], dtype=torch.long)
    answer_ids = torch.tensor(row["answer_input_ids"], dtype=torch.long)
    target_device = get_model_device(target)

    scores, drafter_score_s = cuda_time_call(lambda: score_prompt_blocks(drafter, prompt_ids, args))
    selected = select_blocks_from_scores(
        scores,
        prompt_len=prompt_ids.numel(),
        block_size=args.block_size,
        keep_ratio=args.keep_ratio,
        sink_tokens=args.sink_tokens,
        window_tokens=args.window_tokens,
    )
    compressed_prompt = compress_prompt_by_blocks(prompt_ids, selected, args.block_size)

    full_ids, full_attention, full_labels = _make_answer_batch(prompt_ids, answer_ids, target_device)
    comp_ids, comp_attention, comp_labels = _make_answer_batch(compressed_prompt, answer_ids, target_device)
    full_nll, full_tokens = answer_nll(target, full_ids, full_attention, full_labels)
    comp_nll, comp_tokens = answer_nll(target, comp_ids, comp_attention, comp_labels)

    full_ttft = measure_ttft(target, tokenizer, prompt_ids, args)
    compressed_ttft = measure_ttft(target, tokenizer, compressed_prompt.cpu(), args)

    full_nll_token = (full_nll / full_tokens).item()
    comp_nll_token = (comp_nll / comp_tokens).item()
    nll_delta = comp_nll_token - full_nll_token
    compressed_e2e_ttft = drafter_score_s + compressed_ttft
    speedup_target_only = full_ttft / compressed_ttft if compressed_ttft > 0 else float("inf")
    speedup_e2e = full_ttft / compressed_e2e_ttft if compressed_e2e_ttft > 0 else float("inf")
    compression_ratio = compressed_prompt.numel() / max(1, prompt_ids.numel())

    record = {
        "index": idx,
        "benchmark_id": row.get("benchmark_id", idx),
        "prompt_tokens": int(prompt_ids.numel()),
        "compressed_prompt_tokens": int(compressed_prompt.numel()),
        "compression_ratio": compression_ratio,
        "selected_blocks": selected,
        "num_selected_blocks": len(selected),
        "full_ttft_s": full_ttft,
        "drafter_score_s": drafter_score_s,
        "compressed_ttft_s": compressed_ttft,
        "compressed_e2e_ttft_s": compressed_e2e_ttft,
        "ttft_speedup_target_only": speedup_target_only,
        "ttft_speedup_e2e": speedup_e2e,
        "full_answer_nll_per_token": full_nll_token,
        "compressed_answer_nll_per_token": comp_nll_token,
        "answer_nll_delta_per_token": nll_delta,
        "retained": nll_delta <= args.retention_nll_delta_threshold,
    }
    add_generation_metrics(
        record,
        target=target,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        compressed_prompt=compressed_prompt,
        answer_text=row.get("answer_text", ""),
        args=args,
    )
    return record


def add_generation_metrics(record: Dict, target, tokenizer, prompt_ids, compressed_prompt, answer_text, args) -> None:
    if not args.eval_generation or not answer_text:
        return
    full_pred = generate_answer(target, tokenizer, prompt_ids, args)
    compressed_pred = generate_answer(target, tokenizer, compressed_prompt.cpu(), args)
    record.update(
        {
            "reference_answer": answer_text,
            "full_prediction": full_pred,
            "compressed_prediction": compressed_pred,
            "full_exact_match": exact_match(full_pred, answer_text),
            "compressed_exact_match": exact_match(compressed_pred, answer_text),
            "full_contains_answer": contains_answer(full_pred, answer_text),
            "compressed_contains_answer": contains_answer(compressed_pred, answer_text),
            "full_f1": token_f1(full_pred, answer_text),
            "compressed_f1": token_f1(compressed_pred, answer_text),
        }
    )


def summarize(records: List[Dict]) -> Dict:
    if not records:
        return {}

    def mean(key):
        return sum(float(r[key]) for r in records) / len(records)

    def mean_if_present(key):
        values = [float(r[key]) for r in records if key in r]
        return sum(values) / len(values) if values else None

    full_total = sum(float(r["full_ttft_s"]) for r in records)
    comp_total = sum(float(r["compressed_ttft_s"]) for r in records)
    comp_e2e_total = sum(float(r["compressed_e2e_ttft_s"]) for r in records)
    summary = {
        "num_samples": len(records),
        "avg_prompt_tokens": mean("prompt_tokens"),
        "avg_compressed_prompt_tokens": mean("compressed_prompt_tokens"),
        "avg_compression_ratio": mean("compression_ratio"),
        "avg_full_ttft_s": mean("full_ttft_s"),
        "avg_compressed_ttft_s": mean("compressed_ttft_s"),
        "ttft_speedup_target_only_global": full_total / comp_total if comp_total > 0 else float("inf"),
        "ttft_speedup_e2e_global": full_total / comp_e2e_total if comp_e2e_total > 0 else float("inf"),
        "avg_drafter_score_s": mean("drafter_score_s"),
        "avg_compressed_e2e_ttft_s": mean("compressed_e2e_ttft_s"),
        "avg_ttft_speedup_target_only_per_sample": mean("ttft_speedup_target_only"),
        "avg_ttft_speedup_e2e_per_sample": mean("ttft_speedup_e2e"),
        "avg_full_answer_nll_per_token": mean("full_answer_nll_per_token"),
        "avg_compressed_answer_nll_per_token": mean("compressed_answer_nll_per_token"),
        "avg_answer_nll_delta_per_token": mean("answer_nll_delta_per_token"),
        "retention_rate": sum(1 for r in records if r["retained"]) / len(records),
    }
    for key in (
        "full_exact_match",
        "compressed_exact_match",
        "full_contains_answer",
        "compressed_contains_answer",
        "full_f1",
        "compressed_f1",
    ):
        value = mean_if_present(key)
        if value is not None:
            summary[f"avg_{key}"] = value
    if "avg_full_exact_match" in summary:
        summary["exact_match_delta"] = summary["avg_compressed_exact_match"] - summary["avg_full_exact_match"]
        summary["contains_answer_delta"] = (
            summary["avg_compressed_contains_answer"] - summary["avg_full_contains_answer"]
        )
        summary["f1_delta"] = summary["avg_compressed_f1"] - summary["avg_full_f1"]
    return summary


def main():
    args = get_args()
    dtype = resolve_dtype(args.torch_dtype)
    tokenizer = load_tokenizer(args.tokenizer_id or args.target_model_id)
    dataset = load_eval_rows(args, tokenizer)

    drafter = load_drafter(args, dtype)
    target = load_target(args, dtype)

    Path(os.path.dirname(args.output_jsonl) or ".").mkdir(parents=True, exist_ok=True)
    records = []
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for idx, row in enumerate(tqdm(dataset, desc="SpecPrefill eval")):
            record = evaluate_one(idx, row, drafter, target, tokenizer, args)
            records.append(record)
            out.write(json.dumps(record) + "\n")
            out.flush()

    summary = summarize(records)
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
