import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
import torch


DEFAULT_SFT_SOURCES = [
    "HuggingFaceH4/ultrachat_200k::train_sft::0.70",
    "anon8231489123/ShareGPT_Vicuna_unfiltered::train::0.30",
]


@dataclass(frozen=True)
class SFTSource:
    dataset_id: str
    config: Optional[str]
    split: str
    weight: float


def parse_sft_source(spec: str) -> SFTSource:
    """Parse dataset specs of the form dataset[::config][::split][::weight]."""
    parts = spec.split("::")
    if len(parts) == 1:
        return SFTSource(parts[0], None, "train", 1.0)
    if len(parts) == 2:
        return SFTSource(parts[0], None, parts[1] or "train", 1.0)
    if len(parts) == 3:
        return SFTSource(parts[0], None, parts[1] or "train", float(parts[2] or 1.0))
    if len(parts) == 4:
        return SFTSource(parts[0], parts[1] or None, parts[2] or "train", float(parts[3] or 1.0))
    raise ValueError(f"Invalid SFT source spec: {spec}")


def _load_one_source(source: SFTSource, cache_dir: Optional[str] = None):
    path = Path(source.dataset_id)
    if path.exists():
        ds = load_from_disk(str(path))
    elif source.config:
        ds = load_dataset(source.dataset_id, source.config, split=source.split, cache_dir=cache_dir)
    else:
        ds = load_dataset(source.dataset_id, split=source.split, cache_dir=cache_dir)

    if isinstance(ds, DatasetDict):
        if source.split not in ds:
            raise ValueError(f"Split '{source.split}' not found in {source.dataset_id}; got {list(ds.keys())}")
        ds = ds[source.split]
    return ds


def _get_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _role_of(message: Dict[str, Any]) -> str:
    raw = _get_text(message.get("role", message.get("from", message.get("speaker", "")))).lower()
    if raw in {"human", "user", "prompter"}:
        return "user"
    if raw in {"gpt", "assistant", "bot", "chatgpt"}:
        return "assistant"
    if raw == "system":
        return "system"
    return raw or "user"


def _content_of(message: Dict[str, Any]) -> str:
    for key in ("content", "value", "text", "message"):
        if key in message:
            return _get_text(message[key])
    return ""


def _messages_from_row(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    raw_messages = None
    for key in ("messages", "conversations", "conversation"):
        if isinstance(row.get(key), list):
            raw_messages = row[key]
            break

    if raw_messages:
        messages = []
        for raw in raw_messages:
            if not isinstance(raw, dict):
                continue
            content = _content_of(raw)
            if not content:
                continue
            messages.append({"role": _role_of(raw), "content": content})
        return messages or None

    prompt = _get_text(row.get("prompt", row.get("instruction", row.get("question", ""))))
    answer = _get_text(
        row.get(
            "response",
            row.get("answer", row.get("output", row.get("completion", ""))),
        )
    )
    if prompt and answer:
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]

    chosen = row.get("chosen")
    if isinstance(chosen, list):
        messages = []
        for raw in chosen:
            if isinstance(raw, dict):
                content = _content_of(raw)
                if content:
                    messages.append({"role": _role_of(raw), "content": content})
        return messages or None

    return None


def split_prompt_answer(row: Dict[str, Any]) -> Dict[str, Any]:
    messages = _messages_from_row(row)
    if not messages:
        return {"prompt_messages": [], "answer": "", "valid": False}

    last_assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx]["role"] == "assistant" and messages[idx]["content"]:
            last_assistant_idx = idx
            break
    if last_assistant_idx is None or last_assistant_idx == 0:
        return {"prompt_messages": [], "answer": "", "valid": False}

    prompt_messages = messages[:last_assistant_idx]
    answer = messages[last_assistant_idx]["content"]
    has_user = any(m["role"] == "user" and m["content"] for m in prompt_messages)
    return {
        "prompt_messages": prompt_messages,
        "answer": answer,
        "valid": bool(has_user and answer),
    }


def _fallback_chat_prompt(messages: Sequence[Dict[str, str]]) -> str:
    chunks = []
    for message in messages:
        role = message["role"].strip().capitalize() or "User"
        chunks.append(f"### {role}:\n{message['content'].strip()}")
    chunks.append("### Assistant:\n")
    return "\n\n".join(chunks)


def render_prompt(tokenizer, prompt_messages: Sequence[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                list(prompt_messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return _fallback_chat_prompt(prompt_messages)


def tokenize_prompt_answer(
    row: Dict[str, Any],
    tokenizer,
    max_prompt_tokens: int,
    max_answer_tokens: int,
    add_eos_to_answer: bool = True,
) -> Dict[str, Any]:
    prompt_text = render_prompt(tokenizer, row["prompt_messages"])
    answer_text = row["answer"].strip()

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_tokens,
    )["input_ids"]
    answer_ids = tokenizer(
        answer_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_answer_tokens,
    )["input_ids"]
    if add_eos_to_answer and tokenizer.eos_token_id is not None:
        if not answer_ids or answer_ids[-1] != tokenizer.eos_token_id:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {
        "prompt_input_ids": prompt_ids,
        "answer_input_ids": answer_ids,
        "input_ids": input_ids,
        "labels": labels,
        "prompt_len": len(prompt_ids),
        "answer_len": len(answer_ids),
    }


def _tokenizer_cache_key(tokenizer, args) -> str:
    payload = {
        "tokenizer": getattr(tokenizer, "name_or_path", repr(tokenizer)),
        "sources": list(getattr(args, "sft_sources", DEFAULT_SFT_SOURCES)),
        "max_prompt_tokens": getattr(args, "max_prompt_tokens", 4096),
        "max_answer_tokens": getattr(args, "max_answer_tokens", 512),
        "max_sft_samples": getattr(args, "max_sft_samples", 950000),
        "seed": getattr(args, "seed", 42),
    }
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:10]


def prepare_sprefill_sft_dataset(args, tokenizer) -> Dataset:
    prepared = getattr(args, "prepared_sft_path", None)
    if prepared:
        return load_from_disk(prepared)

    cache_root = Path(getattr(args, "sft_cache_root", None) or getattr(args, "data_root", "./data")) / "sprefill_sft"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / _tokenizer_cache_key(tokenizer, args)
    if getattr(args, "dataset_prepared", True) and cache_path.exists():
        return load_from_disk(str(cache_path))

    sources = [parse_sft_source(spec) for spec in getattr(args, "sft_sources", DEFAULT_SFT_SOURCES)]
    total_weight = sum(max(src.weight, 0.0) for src in sources) or 1.0
    max_samples = int(getattr(args, "max_sft_samples", 950000))
    seed = int(getattr(args, "seed", 42))

    processed = []
    for idx, source in enumerate(sources):
        raw = _load_one_source(source, cache_dir=getattr(args, "hf_cache_dir", None))
        share = source.weight / total_weight
        source_cap = max(1, int(max_samples * share))
        if len(raw) > source_cap:
            raw = raw.shuffle(seed=seed + idx).select(range(source_cap))

        column_names = list(raw.column_names)
        normalized = raw.map(
            split_prompt_answer,
            remove_columns=column_names,
            desc=f"Normalize {source.dataset_id}",
        ).filter(lambda row: row["valid"], desc=f"Filter {source.dataset_id}")

        tokenized = normalized.map(
            lambda row: tokenize_prompt_answer(
                row,
                tokenizer=tokenizer,
                max_prompt_tokens=getattr(args, "max_prompt_tokens", 4096),
                max_answer_tokens=getattr(args, "max_answer_tokens", 512),
                add_eos_to_answer=getattr(args, "add_eos_to_answer", True),
            ),
            remove_columns=normalized.column_names,
            desc=f"Tokenize {source.dataset_id}",
        )
        tokenized = tokenized.filter(
            lambda row: row["prompt_len"] >= getattr(args, "min_prompt_tokens", 32)
            and row["answer_len"] >= getattr(args, "min_answer_tokens", 4),
            desc=f"Length filter {source.dataset_id}",
        )
        processed.append(tokenized)

    if not processed:
        raise RuntimeError("No SFT samples survived normalization/tokenization.")

    dataset = concatenate_datasets(processed).shuffle(seed=seed)
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    dataset.save_to_disk(str(cache_path))
    return dataset


class SpecPrefillDataCollator:
    def __init__(self, tokenizer, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.label_pad_token_id = label_pad_token_id

    def _pad(self, values: List[List[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(v) for v in values)
        padded = [v + [pad_value] * (max_len - len(v)) for v in values]
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompt_ids = [list(f["prompt_input_ids"]) for f in features]
        answer_ids = [list(f["answer_input_ids"]) for f in features]
        full_ids = [list(f["input_ids"]) for f in features]
        labels = [list(f["labels"]) for f in features]

        prompt_input_ids = self._pad(prompt_ids, self.pad_token_id)
        answer_input_ids = self._pad(answer_ids, self.pad_token_id)
        input_ids = self._pad(full_ids, self.pad_token_id)
        labels_tensor = self._pad(labels, self.label_pad_token_id)

        prompt_lens = torch.tensor([len(v) for v in prompt_ids], dtype=torch.long)
        answer_lens = torch.tensor([len(v) for v in answer_ids], dtype=torch.long)
        full_lens = torch.tensor([len(v) for v in full_ids], dtype=torch.long)

        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": (prompt_input_ids != self.pad_token_id).long(),
            "answer_input_ids": answer_input_ids,
            "answer_attention_mask": (answer_input_ids != self.pad_token_id).long(),
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.pad_token_id).long(),
            "labels": labels_tensor,
            "prompt_lens": prompt_lens,
            "answer_lens": answer_lens,
            "full_lens": full_lens,
        }
