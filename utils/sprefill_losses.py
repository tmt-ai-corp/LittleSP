from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


Block = Tuple[int, int]


@dataclass
class CompressionResult:
    input_ids: torch.Tensor
    selected_blocks: List[int]
    num_prompt_tokens: int
    num_compressed_tokens: int


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_blocks(length: int, block_size: int) -> List[Block]:
    if length <= 0:
        return []
    return [(start, min(start + block_size, length)) for start in range(0, length, block_size)]


def select_evenly(num_items: int, max_items: Optional[int]) -> List[int]:
    if max_items is None or max_items <= 0 or num_items <= max_items:
        return list(range(num_items))
    if max_items == 1:
        return [num_items - 1]
    raw = torch.linspace(0, num_items - 1, steps=max_items)
    return sorted(set(int(round(v.item())) for v in raw))


def _outputs_logits(outputs):
    return outputs.logits if hasattr(outputs, "logits") else outputs["logits"]


@torch.no_grad()
def answer_nll(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = _outputs_logits(outputs)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    nlls = []
    counts = []
    for sample_logits, sample_labels, sample_mask in zip(shift_logits, shift_labels, mask):
        selected_logits = sample_logits[sample_mask].float()
        selected_labels = sample_labels[sample_mask]
        count = selected_labels.numel()
        if count:
            nll = F.cross_entropy(selected_logits, selected_labels, reduction="sum")
        else:
            nll = selected_logits.new_zeros(())
        nlls.append(nll)
        counts.append(count)
    return (
        torch.stack(nlls),
        torch.tensor(counts, dtype=torch.long, device=logits.device).clamp_min(1),
    )


def student_answer_ce(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = student_logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def answer_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    shift_s = student_logits[:, :-1, :].contiguous().float() / temperature
    shift_t = teacher_logits[:, :-1, :].contiguous().float() / temperature
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    log_p_s = F.log_softmax(shift_s, dim=-1)
    p_t = F.softmax(shift_t, dim=-1)
    per_token = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)
    denom = mask.sum().clamp_min(1)
    return (per_token * mask).sum() / denom * (temperature ** 2)


def _pad_variants(
    ids: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(x.numel() for x in ids)
    padded_ids = []
    padded_labels = []
    attention = []
    for seq, lab in zip(ids, labels):
        pad = max_len - seq.numel()
        padded_ids.append(F.pad(seq, (0, pad), value=pad_token_id))
        padded_labels.append(F.pad(lab, (0, pad), value=-100))
        attention.append(F.pad(torch.ones_like(seq), (0, pad), value=0))
    return (
        torch.stack(padded_ids).to(device),
        torch.stack(attention).to(device),
        torch.stack(padded_labels).to(device),
    )


@torch.no_grad()
def teacher_deletion_utility(
    teacher_model,
    batch: Dict[str, torch.Tensor],
    block_size: int,
    pad_token_id: int,
    max_oracle_blocks: Optional[int] = 16,
    oracle_microbatch_size: int = 1,
    clamp_min: float = 0.0,
    return_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-block utility labels: NLL(x without block) - NLL(full x)."""
    teacher_device = get_model_device(teacher_model)
    return_device = return_device or batch["input_ids"].device
    prompt_lens = batch["prompt_lens"].detach().cpu().tolist()
    answer_lens = batch["answer_lens"].detach().cpu().tolist()
    max_blocks = max((math.ceil(max(plen, 1) / block_size) for plen in prompt_lens), default=1)

    utilities = torch.zeros((len(prompt_lens), max_blocks), dtype=torch.float32, device=return_device)
    utility_mask = torch.zeros((len(prompt_lens), max_blocks), dtype=torch.bool, device=return_device)

    for b_idx, (prompt_len, answer_len) in enumerate(zip(prompt_lens, answer_lens)):
        prompt_ids = batch["prompt_input_ids"][b_idx, :prompt_len].detach().to("cpu")
        answer_ids = batch["answer_input_ids"][b_idx, :answer_len].detach().to("cpu")
        blocks = split_blocks(prompt_len, block_size)
        selected = select_evenly(len(blocks), max_oracle_blocks)
        if not blocks or not selected or answer_ids.numel() == 0:
            continue

        full_ids = torch.cat([prompt_ids, answer_ids], dim=0)
        full_labels = torch.cat(
            [torch.full_like(prompt_ids, -100), answer_ids],
            dim=0,
        )
        variant_ids = [full_ids]
        variant_labels = [full_labels]
        for block_idx in selected:
            start, end = blocks[block_idx]
            kept_prompt = torch.cat([prompt_ids[:start], prompt_ids[end:]], dim=0)
            ids = torch.cat([kept_prompt, answer_ids], dim=0)
            labels = torch.cat([torch.full_like(kept_prompt, -100), answer_ids], dim=0)
            variant_ids.append(ids)
            variant_labels.append(labels)

        nll_chunks = []
        microbatch_size = max(1, int(oracle_microbatch_size))
        for start in range(0, len(variant_ids), microbatch_size):
            ids, attention, labels = _pad_variants(
                variant_ids[start:start + microbatch_size],
                variant_labels[start:start + microbatch_size],
                pad_token_id,
                teacher_device,
            )
            chunk_nll, _ = answer_nll(teacher_model, ids, attention, labels)
            nll_chunks.append(chunk_nll.detach().cpu())
            del ids, attention, labels, chunk_nll
        nll = torch.cat(nll_chunks, dim=0)
        base = nll[0]
        deltas = (nll[1:] - base).float().clamp_min(clamp_min)

        for local_idx, block_idx in enumerate(selected):
            utilities[b_idx, block_idx] = deltas[local_idx].to(return_device)
            utility_mask[b_idx, block_idx] = True

    return utilities, utility_mask


def attention_looks_like_probabilities(
    attentions: Sequence[torch.Tensor],
    batch_size: int,
    max_prompt_len: int,
) -> bool:
    """Cheaply reject hidden-state-like or non-finite tensors mislabeled as attention."""
    for raw_attn in reversed(attentions):
        if raw_attn is None:
            continue
        try:
            attn = normalize_attention_shape(raw_attn, batch_size=batch_size)
        except (TypeError, ValueError):
            return False
        if attn.size(-2) < max_prompt_len or attn.size(-1) < max_prompt_len:
            return False
        query_count = min(4, max_prompt_len)
        sample = attn[0, 0, max_prompt_len - query_count:max_prompt_len, :max_prompt_len].detach().float()
        if sample.numel() == 0 or not torch.isfinite(sample).all():
            return False
        if sample.min() < -1e-4 or sample.max() > 2.0:
            return False
        row_sums = sample.sum(dim=-1)
        return bool(((row_sums > 0.1) & (row_sums < 2.0)).all().item())
    return False


def normalize_attention_shape(attn: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Normalize common attention layouts to [batch, heads, query, key]."""
    if not isinstance(attn, torch.Tensor):
        raise TypeError(f"Expected an attention tensor, got {type(attn).__name__}.")

    if attn.dim() == 5:
        if attn.size(0) != batch_size:
            raise ValueError(
                f"Cannot interpret rank-5 attention shape {tuple(attn.shape)} "
                f"for batch size {batch_size}."
            )
        return attn.flatten(1, 2)

    if attn.dim() == 4:
        if attn.size(0) != batch_size:
            raise ValueError(
                f"Cannot interpret rank-4 attention shape {tuple(attn.shape)} "
                f"for batch size {batch_size}."
            )
        return attn

    if attn.dim() == 3:
        first_dim, query_len, key_len = attn.shape
        if first_dim == batch_size:
            # Head-aggregated attention: [batch, query, key].
            return attn.unsqueeze(1)
        if first_dim % batch_size == 0:
            # Flattened heads: [batch * heads, query, key].
            return attn.reshape(batch_size, first_dim // batch_size, query_len, key_len)
        raise ValueError(
            f"Cannot interpret rank-3 attention shape {tuple(attn.shape)} "
            f"for batch size {batch_size}."
        )

    if attn.dim() == 2 and batch_size == 1:
        # Head- and batch-aggregated attention: [query, key].
        return attn.unsqueeze(0).unsqueeze(0)

    raise ValueError(
        f"Unsupported attention shape {tuple(attn.shape)} for batch size {batch_size}; "
        "expected [q,k], [b,q,k], [b*h,q,k], [b,h,q,k], or [b,g,h,q,k]."
    )


def aggregate_attention_block_scores(
    attentions: Sequence[torch.Tensor],
    prompt_lens: torch.Tensor,
    block_size: int,
    score_query_tokens: int = 128,
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
    aggregation: str = "max_mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not attentions:
        raise ValueError("Student model did not return attentions. Use eager attention and output_attentions=True.")

    raw_layer_slice = attentions[slice(layer_start, layer_end)]
    if not raw_layer_slice:
        raw_layer_slice = attentions[-1:]

    prompt_lens_cpu = prompt_lens.detach().cpu().tolist()
    batch_size = len(prompt_lens_cpu)
    layer_slice = [
        normalize_attention_shape(attn, batch_size=batch_size)
        for attn in raw_layer_slice
        if attn is not None
    ]
    if not layer_slice:
        raise ValueError("Student model returned only empty attention entries.")

    device = layer_slice[0].device
    rows = []
    masks = []

    for batch_idx, prompt_len in enumerate(prompt_lens_cpu):
        blocks = split_blocks(prompt_len, block_size)
        if not blocks:
            rows.append(torch.zeros(1, device=device))
            masks.append(torch.zeros(1, dtype=torch.bool, device=device))
            continue

        q_start = max(0, prompt_len - score_query_tokens)
        q_end = prompt_len
        if q_end <= q_start:
            q_start = max(0, prompt_len - 1)

        if aggregation == "mean":
            layer_token_scores = []
            for attn in layer_slice:
                part = torch.nan_to_num(
                    attn[batch_idx, :, q_start:q_end, :prompt_len].float()
                )
                layer_token_scores.append(part.mean(dim=(0, 1)))
            token_scores = torch.stack(layer_token_scores, dim=0).mean(dim=0)
        elif aggregation == "last_mean":
            part = torch.nan_to_num(
                layer_slice[-1][batch_idx, :, q_start:q_end, :prompt_len].float()
            )
            token_scores = part.mean(dim=(0, 1))
        else:
            per_layer = []
            for attn in layer_slice:
                part = torch.nan_to_num(
                    attn[batch_idx, :, q_start:q_end, :prompt_len].float()
                )
                per_layer.append(part.amax(dim=0))
            token_scores = torch.stack(per_layer, dim=0).amax(dim=0).mean(dim=0)

        if token_scores.numel() < prompt_len:
            raise ValueError(
                f"Attention key dimension {token_scores.numel()} is shorter than prompt length "
                f"{prompt_len}; the model likely returned hidden states instead of attention probabilities."
            )
        block_scores = [token_scores[start:end].mean() for start, end in blocks]
        rows.append(torch.stack(block_scores))
        masks.append(torch.ones(len(block_scores), dtype=torch.bool, device=device))

    max_blocks = max(row.numel() for row in rows)
    padded_rows = []
    padded_masks = []
    for row, mask in zip(rows, masks):
        pad = max_blocks - row.numel()
        padded_rows.append(F.pad(row, (0, pad), value=-1e9))
        padded_masks.append(F.pad(mask, (0, pad), value=False))
    return torch.stack(padded_rows), torch.stack(padded_masks)


def aggregate_hidden_block_scores(
    hidden_states: Sequence[torch.Tensor],
    prompt_lens: torch.Tensor,
    block_size: int,
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
    aggregation: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not hidden_states:
        raise ValueError("Student model did not return hidden states.")

    layer_slice = hidden_states[slice(layer_start, layer_end)]
    if not layer_slice:
        layer_slice = hidden_states[-1:]

    device = layer_slice[0].device
    rows = []
    masks = []
    for batch_idx, prompt_len in enumerate(prompt_lens.detach().cpu().tolist()):
        blocks = split_blocks(prompt_len, block_size)
        per_layer = []
        for hidden in layer_slice:
            values = torch.nan_to_num(hidden[batch_idx, :prompt_len].float())
            per_layer.append(values.square().mean(dim=-1).clamp_min(1e-12).sqrt())
        stacked = torch.stack(per_layer, dim=0)
        if aggregation == "last_mean":
            token_scores = stacked[-1]
        elif aggregation == "max_mean":
            token_scores = stacked.amax(dim=0)
        else:
            token_scores = stacked.mean(dim=0)

        block_scores = [token_scores[start:end].mean() for start, end in blocks]
        rows.append(torch.stack(block_scores))
        masks.append(torch.ones(len(block_scores), dtype=torch.bool, device=device))

    max_blocks = max(row.numel() for row in rows)
    return (
        torch.stack([F.pad(row, (0, max_blocks - row.numel()), value=-1e9) for row in rows]),
        torch.stack([F.pad(mask, (0, max_blocks - mask.numel()), value=False) for mask in masks]),
    )


def saliency_kl_loss(
    student_scores: torch.Tensor,
    teacher_utility: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    valid_rows = mask.sum(dim=1).gt(1)
    if not valid_rows.any():
        return student_scores.new_zeros(())
    s = student_scores[valid_rows].masked_fill(~mask[valid_rows], -1e9)
    u = teacher_utility[valid_rows].masked_fill(~mask[valid_rows], -1e9)
    target = F.softmax(u / temperature, dim=-1)
    log_pred = F.log_softmax(s / temperature, dim=-1)
    return F.kl_div(log_pred, target, reduction="batchmean") * (temperature ** 2)


def pairwise_rank_loss(
    student_scores: torch.Tensor,
    teacher_utility: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    valid_rows = mask.sum(dim=1).gt(1)
    if not valid_rows.any():
        return student_scores.new_zeros(())

    s = student_scores[valid_rows]
    u = teacher_utility[valid_rows]
    m = mask[valid_rows]
    delta_u = u.unsqueeze(2) - u.unsqueeze(1)
    delta_s = s.unsqueeze(2) - s.unsqueeze(1)
    pair_mask = m.unsqueeze(2) & m.unsqueeze(1) & delta_u.abs().gt(margin)
    if not pair_mask.any():
        return student_scores.new_zeros(())
    target = delta_u.sign()
    weights = delta_u.abs().detach()
    loss = F.softplus(-target * delta_s) * weights
    return loss[pair_mask].mean()


def budget_loss(
    student_scores: torch.Tensor,
    mask: torch.Tensor,
    keep_ratio: float,
    temperature: float = 0.1,
) -> torch.Tensor:
    losses = []
    for row, row_mask in zip(student_scores, mask):
        valid = row[row_mask]
        if valid.numel() <= 1:
            continue
        k = max(1, int(math.ceil(valid.numel() * keep_ratio)))
        threshold = torch.topk(valid.detach(), k=k).values[-1]
        gate = torch.sigmoid((valid - threshold) / temperature)
        losses.append((gate.mean() - keep_ratio) ** 2)
    if not losses:
        return student_scores.new_zeros(())
    return torch.stack(losses).mean()


def select_blocks_from_scores(
    scores: torch.Tensor,
    prompt_len: int,
    block_size: int,
    keep_ratio: float,
    sink_tokens: int = 0,
    window_tokens: int = 0,
) -> List[int]:
    blocks = split_blocks(prompt_len, block_size)
    if not blocks:
        return []
    num_blocks = len(blocks)
    k = max(1, int(math.ceil(num_blocks * keep_ratio)))
    forced = set()
    for idx, (start, end) in enumerate(blocks):
        if sink_tokens and start < sink_tokens:
            forced.add(idx)
        if window_tokens and end > max(0, prompt_len - window_tokens):
            forced.add(idx)

    available = [idx for idx in range(num_blocks) if idx not in forced]
    need = max(0, k - len(forced))
    selected = set(forced)
    if need > 0 and available:
        values = scores[:num_blocks].detach().float()
        ranked = sorted(available, key=lambda idx: values[idx].item(), reverse=True)
        selected.update(ranked[:need])
    return sorted(selected)


def compress_prompt_by_blocks(
    prompt_ids: torch.Tensor,
    selected_blocks: Sequence[int],
    block_size: int,
) -> torch.Tensor:
    blocks = split_blocks(prompt_ids.numel(), block_size)
    pieces = []
    for idx in selected_blocks:
        if 0 <= idx < len(blocks):
            start, end = blocks[idx]
            pieces.append(prompt_ids[start:end])
    if not pieces:
        return prompt_ids[-min(prompt_ids.numel(), block_size):]
    return torch.cat(pieces, dim=0)


def cuda_time_call(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start
