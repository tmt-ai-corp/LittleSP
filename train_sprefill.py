import argparse
import datetime
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, set_seed

from quantization.utils import apply_littlebit_patch
from utils.datautils import load_tokenizer
from utils.misc import setup_logger
from utils.sprefill_data import (
    DEFAULT_SFT_SOURCES,
    SpecPrefillDataCollator,
    prepare_sprefill_sft_dataset,
)
from utils.sprefill_trainer import SpecPrefillKDTrainer, SpecPrefillLossConfig
from utils.utils import prepare_model_for_training, print_trainable_parameters


logger = setup_logger(__name__)


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
        description="Train a LittleBit drafter for Speculative Prefill block filtering."
    )

    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--teacher_model_id", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./outputs_sprefill")
    parser.add_argument("--f_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sft_sources", nargs="+", default=DEFAULT_SFT_SOURCES)
    parser.add_argument("--prepared_sft_path", type=str, default=None)
    parser.add_argument("--sft_cache_root", type=str, default="./data")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--dataset_prepared", type=str2bool, default=True)
    parser.add_argument("--max_sft_samples", type=int, default=950000)
    parser.add_argument("--max_prompt_tokens", type=int, default=4096)
    parser.add_argument("--max_answer_tokens", type=int, default=512)
    parser.add_argument("--min_prompt_tokens", type=int, default=32)
    parser.add_argument("--min_answer_tokens", type=int, default=4)
    parser.add_argument("--add_eos_to_answer", type=str2bool, default=True)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--report", nargs="+", default=["tensorboard"], choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--run_name", type=str, default="sprefill_littlebit")
    parser.add_argument("--ds_config_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)

    parser.add_argument("--quant_func", type=str, default="SmoothSign")
    parser.add_argument("--quant_mod", type=str, default="LittleBitLinear")
    parser.add_argument("--residual", type=str2bool, default=False)
    parser.add_argument("--split_dim", type=int, default=1024)
    parser.add_argument("--eff_bit", type=float, default=0.1)
    parser.add_argument("--kv_factor", type=float, default=1.0)
    parser.add_argument("--min_split_dim", type=int, default=8)
    parser.add_argument("--use_itq", type=str2bool, default=False)
    parser.add_argument("--itq_n_iter", type=int, default=50)

    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--teacher_device_map", type=str, default="auto")
    parser.add_argument("--student_device", type=str, default=None)

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_oracle_blocks", type=int, default=16)
    parser.add_argument("--score_query_tokens", type=int, default=128)
    parser.add_argument("--score_layer_start", type=int, default=None)
    parser.add_argument("--score_layer_end", type=int, default=None)
    parser.add_argument("--score_aggregation", type=str, default="max_mean", choices=["max_mean", "mean", "last_mean"])
    parser.add_argument("--keep_ratio", type=float, default=0.05)
    parser.add_argument("--saliency_temperature", type=float, default=0.1)
    parser.add_argument("--budget_temperature", type=float, default=0.1)
    parser.add_argument("--utility_clamp_min", type=float, default=0.0)
    parser.add_argument("--pairwise_margin", type=float, default=0.0)
    parser.add_argument("--rank_loss_scale", type=float, default=1.0)
    parser.add_argument("--pairwise_loss_scale", type=float, default=0.25)
    parser.add_argument("--budget_loss_scale", type=float, default=0.05)
    parser.add_argument("--answer_ce_loss_scale", type=float, default=0.0)
    parser.add_argument("--logit_kd_loss_scale", type=float, default=0.0)
    parser.add_argument("--hidden_mse_loss_scale", type=float, default=0.0)
    parser.add_argument("--kd_temperature", type=float, default=1.0)

    return parser.parse_args()


def get_save_dir(args) -> str:
    name = args.f_name or datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_sprefill")
    save_dir = os.path.join(args.save_dir, name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir


def get_torch_dtype(args):
    if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_student_model(args, torch_dtype):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.config.output_attentions = True
    if args.gradient_checkpointing:
        prepare_model_for_training(model)
    else:
        for name, param in model.named_parameters():
            if "lm_head" in name or "embed" in name:
                param.requires_grad = False

    logger.info("Applying LittleBitLinear patch to student.")
    model = apply_littlebit_patch(model, args, do_train=True)

    if args.student_device:
        model.to(torch.device(args.student_device))
    elif torch.cuda.is_available() and args.ds_config_path is None:
        model.to(torch.device("cuda"))

    print_trainable_parameters(model)
    return model


def load_teacher_model(args, torch_dtype):
    teacher_id = args.teacher_model_id or args.model_id
    kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.teacher_device_map and args.teacher_device_map.lower() != "none":
        kwargs["device_map"] = args.teacher_device_map
    model = AutoModelForCausalLM.from_pretrained(teacher_id, **kwargs)
    model.eval()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False
    if "device_map" not in kwargs and torch.cuda.is_available():
        model.to(torch.device("cuda"))
    return model


def make_training_args(args, save_dir):
    report_to = [] if args.report == ["none"] else args.report
    return TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        deepspeed=args.ds_config_path,
        report_to=report_to,
        run_name=args.run_name,
        remove_unused_columns=False,
    )


def build_loss_config(args) -> SpecPrefillLossConfig:
    return SpecPrefillLossConfig(
        block_size=args.block_size,
        max_oracle_blocks=args.max_oracle_blocks,
        score_query_tokens=args.score_query_tokens,
        score_layer_start=args.score_layer_start,
        score_layer_end=args.score_layer_end,
        score_aggregation=args.score_aggregation,
        keep_ratio=args.keep_ratio,
        saliency_temperature=args.saliency_temperature,
        budget_temperature=args.budget_temperature,
        utility_clamp_min=args.utility_clamp_min,
        pairwise_margin=args.pairwise_margin,
        rank_loss_scale=args.rank_loss_scale,
        pairwise_loss_scale=args.pairwise_loss_scale,
        budget_loss_scale=args.budget_loss_scale,
        answer_ce_loss_scale=args.answer_ce_loss_scale,
        logit_kd_loss_scale=args.logit_kd_loss_scale,
        hidden_mse_loss_scale=args.hidden_mse_loss_scale,
        kd_temperature=args.kd_temperature,
    )


def save_sprefill_metadata(args, save_dir):
    quant_params = {
        "quant_func": args.quant_func,
        "eff_bit": args.eff_bit,
        "split_dim": args.split_dim,
        "residual": args.residual,
        "kv_factor": args.kv_factor,
        "min_split_dim": args.min_split_dim,
        "quant_mod": args.quant_mod,
        "use_itq": args.use_itq,
        "itq_n_iter": args.itq_n_iter,
    }
    sprefill_params = vars(args).copy()
    with open(os.path.join(save_dir, "littlebit_config.json"), "w", encoding="utf-8") as f:
        json.dump(quant_params, f, indent=2)
    with open(os.path.join(save_dir, "sprefill_config.json"), "w", encoding="utf-8") as f:
        json.dump(sprefill_params, f, indent=2, default=str)


def main():
    args = get_args()
    set_seed(args.seed)
    save_dir = get_save_dir(args)
    torch_dtype = get_torch_dtype(args)

    logger.info("Loading tokenizer.")
    tokenizer = load_tokenizer(args.model_id)

    logger.info("Preparing SFT mixture dataset.")
    train_dataset = prepare_sprefill_sft_dataset(args, tokenizer)
    logger.info(f"Loaded {len(train_dataset)} SFT samples.")

    logger.info("Loading LittleBit student.")
    student = load_student_model(args, torch_dtype)

    logger.info("Loading frozen teacher.")
    teacher = load_teacher_model(args, torch_dtype)

    trainer = SpecPrefillKDTrainer(
        model=student,
        teacher_model=teacher,
        tokenizer=tokenizer,
        loss_config=build_loss_config(args),
        args=make_training_args(args, save_dir),
        train_dataset=train_dataset,
        data_collator=SpecPrefillDataCollator(tokenizer),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    save_sprefill_metadata(args, save_dir)
    logger.info(f"Speculative Prefill LittleBit checkpoint saved to {save_dir}")


if __name__ == "__main__":
    main()
