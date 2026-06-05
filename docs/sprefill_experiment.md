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
  --oracle_microbatch_size 1 \
  --score_source auto \
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

# Trade memory for faster teacher-oracle generation
--oracle_microbatch_size 2

# Force a scorer for an ablation
--score_source attention
--score_source hidden_norm

# Change scoring query window
--score_query_tokens 256
```

`--score_source auto` validates that the model's `output_attentions` tensors are
finite attention probabilities. Some Qwen3/Transformers combinations return a
3-D hidden-state-like tensor in that slot. In that case, training automatically
falls back to differentiable hidden-state block scores instead of producing NaN
block means.

The deletion oracle runs variants in microbatches. Keep
`--oracle_microbatch_size 1` for the lowest memory use with large-vocabulary
models such as Qwen3. Raising it speeds up oracle generation but increases peak
memory. The NLL implementation converts only answer-position logits to FP32.

For Qwen3 training, use SDPA with hidden-state scoring:

```bash
--attn_implementation sdpa \
--score_source hidden_norm
```

For multi-GPU Qwen3 training, use the SpecPrefill-specific ZeRO-1 config and
rank-consistent last-layer scoring:

```bash
deepspeed --num_gpus=4 train_sprefill.py \
  ... \
  --ds_config_path configs/zero_sprefill.json \
  --teacher_device_map local \
  --attn_implementation sdpa \
  --score_source hidden_norm \
  --score_aggregation last_mean
```

The generic `configs/zero.json` uses ZeRO-2 gradient partitioning with
overlapped reduction. SpecPrefill's data-dependent scoring graph can make its
gradient buckets differ across ranks, causing a first-step NCCL collective
timeout. The dedicated config uses a non-overlapped ZeRO-1 reduction path.
Each distributed process also keeps its frozen teacher on its local-rank GPU;
do not use `teacher_device_map=auto` for this data-parallel setup.

`flex_attention` compiles sequence-length-specific Triton kernels. With variable
SFT lengths, Qwen3, gradient checkpointing, and LittleBit QAT, some generated
kernels can request more shared memory than the GPU provides. The training
script therefore converts `flex_attention + auto` to `sdpa + hidden_norm` by
default. `--allow_flex_attention True` is available only for explicit kernel
experiments and may still fail on particular sequence lengths.

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

For paper benchmarks, pass a JSONL file with either raw text fields or tokenized
fields:

```json
{"id": "niah-0001", "prompt": "...long context...", "answer": "needle answer"}
```

or:

```json
{"id": "case-0001", "prompt_input_ids": [1, 2, 3], "answer_input_ids": [4, 5]}
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_sprefill.py \
  --drafter_model_id ./outputs_sprefill/run_name \
  --target_model_id meta-llama/Llama-2-7b-hf \
  --benchmark_jsonl ./benchmarks/niah.jsonl \
  --eval_generation True \
  --generation_max_new_tokens 128 \
  --output_jsonl ./niah_sprefill_records.jsonl \
  --output_summary ./niah_sprefill_summary.json
```

With `--eval_generation True`, the summary also records full/compressed exact
match, answer containment, token F1, and the corresponding compressed-minus-full
score deltas.

## Current Scope

This implementation provides the training/eval objective and a PyTorch attention
scoring path. It does not include a custom FlashPrefill CUDA kernel. If kernel
level drafter acceleration is required, use the recorded `drafter_score_s` as the
number to replace with the sparse kernel timing in the final system benchmark.
