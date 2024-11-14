from typing import TYPE_CHECKING, List, Optional

from ...data import get_dataset, get_template_and_fix_tokenizer, PairwiseDataCollatorWithPadding
from ...extras import logging
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy
from .trainer import ToyRewardModelTrainer

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, ModelArguments, Seq2SeqTrainingArguments

logger = logging.get_logger(__name__)

def run_toyrm(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments", 
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)
    data_collator = PairwiseDataCollatorWithPadding(template=template, pad_to_multiple_of=8, tokenizer=tokenizer)

    # Update arguments
    training_args.remove_unused_columns = False

    # Initialize trainer
    trainer = ToyRewardModelTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeAccuracy(),
        tokenizer=tokenizer,
        **dataset_module
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"]) 