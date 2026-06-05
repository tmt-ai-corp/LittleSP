import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import Trainer

from utils.sprefill_losses import (
    aggregate_attention_block_scores,
    aggregate_hidden_block_scores,
    attention_looks_like_probabilities,
    answer_kl,
    budget_loss,
    get_model_device,
    pairwise_rank_loss,
    saliency_kl_loss,
    student_answer_ce,
    teacher_deletion_utility,
)

logger = logging.getLogger(__name__)


@dataclass
class SpecPrefillLossConfig:
    block_size: int = 128
    max_oracle_blocks: int = 16
    oracle_microbatch_size: int = 1
    score_query_tokens: int = 128
    score_layer_start: Optional[int] = None
    score_layer_end: Optional[int] = None
    score_aggregation: str = "max_mean"
    score_source: str = "auto"
    keep_ratio: float = 0.05
    saliency_temperature: float = 0.1
    budget_temperature: float = 0.1
    utility_clamp_min: float = 0.0
    pairwise_margin: float = 0.0
    rank_loss_scale: float = 1.0
    pairwise_loss_scale: float = 0.25
    budget_loss_scale: float = 0.05
    answer_ce_loss_scale: float = 0.0
    logit_kd_loss_scale: float = 0.0
    hidden_mse_loss_scale: float = 0.0
    kd_temperature: float = 1.0


class SpecPrefillKDTrainer(Trainer):
    def __init__(
        self,
        teacher_model,
        tokenizer,
        loss_config: SpecPrefillLossConfig,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.loss_config = loss_config
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self._warned_attention_fallback = False

    @staticmethod
    def _move_tensor_inputs(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def _teacher_forward_for_kd(self, inputs: Dict[str, torch.Tensor], output_hidden_states: bool):
        teacher_device = get_model_device(self.teacher_model)
        teacher_inputs = {
            "input_ids": inputs["input_ids"].to(teacher_device),
            "attention_mask": inputs["attention_mask"].to(teacher_device),
            "use_cache": False,
            "output_hidden_states": output_hidden_states,
        }
        with torch.no_grad():
            return self.teacher_model(**teacher_inputs)

    def _student_full_forward(self, model, inputs: Dict[str, torch.Tensor], output_hidden_states: bool):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=None,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        cfg = self.loss_config
        inputs = self._move_tensor_inputs(inputs, get_model_device(model))

        teacher_utility, teacher_mask = teacher_deletion_utility(
            self.teacher_model,
            inputs,
            block_size=cfg.block_size,
            pad_token_id=self.pad_token_id,
            max_oracle_blocks=cfg.max_oracle_blocks,
            oracle_microbatch_size=cfg.oracle_microbatch_size,
            clamp_min=cfg.utility_clamp_min,
            return_device=inputs["input_ids"].device,
        )

        prompt_outputs = model(
            input_ids=inputs["prompt_input_ids"],
            attention_mask=inputs["prompt_attention_mask"],
            use_cache=False,
            output_attentions=cfg.score_source != "hidden_norm",
            output_hidden_states=cfg.score_source != "attention",
        )

        use_attention = cfg.score_source == "attention"
        if cfg.score_source == "auto":
            use_attention = attention_looks_like_probabilities(
                getattr(prompt_outputs, "attentions", None) or (),
                batch_size=inputs["prompt_input_ids"].size(0),
                max_prompt_len=int(inputs["prompt_lens"].max().item()),
            )
            if not use_attention and not self._warned_attention_fallback:
                logger.warning(
                    "Student attention output is non-finite or does not look like attention "
                    "probabilities; falling back to hidden-state block scoring."
                )
                self._warned_attention_fallback = True

        if use_attention:
            student_scores, score_mask = aggregate_attention_block_scores(
                prompt_outputs.attentions,
                prompt_lens=inputs["prompt_lens"],
                block_size=cfg.block_size,
                score_query_tokens=cfg.score_query_tokens,
                layer_start=cfg.score_layer_start,
                layer_end=cfg.score_layer_end,
                aggregation=cfg.score_aggregation,
            )
        else:
            student_scores, score_mask = aggregate_hidden_block_scores(
                prompt_outputs.hidden_states,
                prompt_lens=inputs["prompt_lens"],
                block_size=cfg.block_size,
                layer_start=cfg.score_layer_start,
                layer_end=cfg.score_layer_end,
                aggregation=cfg.score_aggregation,
            )

        student_nonfinite = (~torch.isfinite(student_scores[score_mask])).float().mean()
        student_scores = torch.nan_to_num(student_scores, nan=0.0, posinf=1e4, neginf=-1e4)
        teacher_utility = torch.nan_to_num(teacher_utility, nan=0.0, posinf=1e4, neginf=0.0)

        dim = min(student_scores.size(1), teacher_utility.size(1))
        student_scores = student_scores[:, :dim]
        score_mask = score_mask[:, :dim]
        teacher_utility = teacher_utility[:, :dim].to(student_scores.device)
        teacher_mask = teacher_mask[:, :dim].to(student_scores.device)
        joint_mask = score_mask & teacher_mask

        rank_loss = saliency_kl_loss(
            student_scores,
            teacher_utility,
            joint_mask,
            temperature=cfg.saliency_temperature,
        )
        pair_loss = pairwise_rank_loss(
            student_scores,
            teacher_utility,
            joint_mask,
            margin=cfg.pairwise_margin,
        )
        keep_budget_loss = budget_loss(
            student_scores,
            score_mask,
            keep_ratio=cfg.keep_ratio,
            temperature=cfg.budget_temperature,
        )

        loss = (
            cfg.rank_loss_scale * rank_loss
            + cfg.pairwise_loss_scale * pair_loss
            + cfg.budget_loss_scale * keep_budget_loss
        )

        outputs = None
        answer_ce = student_scores.new_zeros(())
        logit_kd = student_scores.new_zeros(())
        hidden_mse = student_scores.new_zeros(())

        need_full_student = (
            cfg.answer_ce_loss_scale > 0
            or cfg.logit_kd_loss_scale > 0
            or cfg.hidden_mse_loss_scale > 0
            or return_outputs
        )
        need_teacher_kd = cfg.logit_kd_loss_scale > 0 or cfg.hidden_mse_loss_scale > 0

        if need_full_student:
            outputs = self._student_full_forward(
                model,
                inputs,
                output_hidden_states=cfg.hidden_mse_loss_scale > 0,
            )
            student_logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            if cfg.answer_ce_loss_scale > 0:
                answer_ce = student_answer_ce(student_logits, inputs["labels"])
                loss = loss + cfg.answer_ce_loss_scale * answer_ce

            if need_teacher_kd:
                teacher_outputs = self._teacher_forward_for_kd(
                    inputs,
                    output_hidden_states=cfg.hidden_mse_loss_scale > 0,
                )
                teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, "logits") else teacher_outputs["logits"]
                teacher_logits = teacher_logits.to(student_logits.device)
                if cfg.logit_kd_loss_scale > 0:
                    logit_kd = answer_kl(
                        student_logits,
                        teacher_logits,
                        inputs["labels"],
                        temperature=cfg.kd_temperature,
                    )
                    loss = loss + cfg.logit_kd_loss_scale * logit_kd

                if cfg.hidden_mse_loss_scale > 0:
                    student_hidden = outputs.hidden_states[1:]
                    teacher_hidden = teacher_outputs.hidden_states[1:]
                    count = min(len(student_hidden), len(teacher_hidden))
                    mse_terms = []
                    for idx in range(count):
                        mse_terms.append(
                            F.mse_loss(
                                student_hidden[idx].float(),
                                teacher_hidden[idx].to(student_hidden[idx].device).float(),
                            )
                        )
                    if mse_terms:
                        hidden_mse = torch.stack(mse_terms).mean()
                        loss = loss + cfg.hidden_mse_loss_scale * hidden_mse

        if not loss.requires_grad:
            raise RuntimeError(
                "Speculative prefill loss is detached from the student model. "
                "Check that the selected score source depends on trainable parameters."
            )

        with torch.no_grad():
            valid_util = teacher_utility[joint_mask]
            mean_utility = valid_util.mean() if valid_util.numel() else teacher_utility.new_zeros(())
            self.log(
                {
                    "loss_rank": rank_loss.detach().float().item(),
                    "loss_pairwise": pair_loss.detach().float().item(),
                    "loss_budget": keep_budget_loss.detach().float().item(),
                    "loss_answer_ce": answer_ce.detach().float().item(),
                    "loss_logit_kd": logit_kd.detach().float().item(),
                    "loss_hidden_mse": hidden_mse.detach().float().item(),
                    "teacher_utility_mean": mean_utility.detach().float().item(),
                    "oracle_blocks_per_sample": joint_mask.sum(dim=1).float().mean().item(),
                    "degenerate_supervision_fraction": (
                        joint_mask.sum(dim=1).le(1).float().mean().item()
                    ),
                    "score_source_attention": float(use_attention),
                    "student_score_nonfinite_fraction": student_nonfinite.detach().float().item(),
                }
            )

        return (loss, outputs if outputs is not None else prompt_outputs) if return_outputs else loss
