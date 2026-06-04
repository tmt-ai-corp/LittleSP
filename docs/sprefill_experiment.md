# Speculative Prefill LittleBit Drafter

This experiment trains a LittleBitLinear drafter for prompt block filtering rather
than for ordinary perplexity or zero-shot accuracy.

## Objective

For an SFT pair `(prompt x, answer a)`, a frozen target model defines block
utility by deletion regret:

```text
u_j = NLL_T(a | x without block_j) - NLL_T(a | x)
```

The LittleBit student sees only the prompt and emits block scores from its
prompt attention maps. Training distills the target utility ranking into those
student scores:

```text
L = lambda_rank * KL(softmax(u/tau), softmax(s/tau))
  + lambda_pair * pairwise_rank(s, u)
  + lambda_budget * (mean(soft_topk_gate(s)) - keep_ratio)^2
  + optional answer CE / logit KD / hidden MSE
```

This makes the checkpoint a Speculative Prefill scorer, not a checkpoint whose
main quality signal is PPL or lm-eval zero-shot.

## SFT Data

Default mixture:

```text
HuggingFaceH4/ultrachat_200k::train_sft::0.70
anon8231489123/ShareGPT_Vicuna_unfiltered::train::0.30
```

The loader caps the total by `--max_sft_samples` and defaults to `950000`, so the
mixture stays under one million examples. For an offline network, prepare the
dataset once with `datasets.save_to_disk()` and pass:

```bash
--prepared_sft_path /path/to/pretokenized_sprefill_sft
```

## Training

Example 0.1 bpp LittleBit drafter:

```bash
CUDA_VISIBLE_DEVICES=0 python train_sprefill.py \
  --model_id meta-llama/Llama-2-7b-hf \
  --teacher_model_id meta-llama/Llama-2-7b-hf \
  --save_dir ./outputs_sprefill \
  --max_sft_samples 950000 \
  --max_prompt_tokens 4096 \
  --max_answer_tokens 512 \
  --quant_mod LittleBitLinear \
  --quant_func SmoothSign \
  --eff_bit 0.1 \
  --residual False \
  --block_size 128 \
  --keep_ratio 0.05 \
  --max_oracle_blocks 16 \
  --rank_loss_scale 1.0 \
  --pairwise_loss_scale 0.25 \
  --budget_loss_scale 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1
```

Useful ablations:

```bash
# Rank-only utility distillation
--pairwise_loss_scale 0 --budget_loss_scale 0

# Add answer CE regularization
--answer_ce_loss_scale 0.05

# Add conventional answer-token logit KD
--logit_kd_loss_scale 0.1 --kd_temperature 2.0

# Increase oracle density
--max_oracle_blocks 32

# Change scoring query window
--score_query_tokens 256
```

## Evaluation

The eval script loads the trained LittleBit drafter, scores prompt blocks, keeps
top blocks plus sink/window blocks, and evaluates the target on both full and
compressed prompts.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_sprefill.py \
  --drafter_model_id ./outputs_sprefill/run_name \
  --target_model_id meta-llama/Llama-2-7b-hf \
  --max_sft_samples 512 \
  --max_prompt_tokens 4096 \
  --max_answer_tokens 512 \
  --block_size 128 \
  --keep_ratio 0.05 \
  --sink_tokens 256 \
  --window_tokens 512 \
  --output_jsonl ./sprefill_eval_records.jsonl \
  --output_summary ./sprefill_eval_summary.json
```

Recorded metrics:

- `ttft_speedup_target_only_global`: target full prompt TTFT divided by target
  compressed prompt TTFT.
- `ttft_speedup_e2e_global`: target full prompt TTFT divided by drafter scoring
  time plus compressed target TTFT.
- `avg_answer_nll_delta_per_token`: compressed answer NLL minus full answer NLL.
- `retention_rate`: fraction of examples with answer NLL delta under
  `--retention_nll_delta_threshold`.

## Current Scope

This implementation provides the training/eval objective and a PyTorch attention
scoring path. It does not include a custom FlashPrefill CUDA kernel. If kernel
level drafter acceleration is required, use the recorded `drafter_score_s` as the
number to replace with the sparse kernel timing in the final system benchmark.
