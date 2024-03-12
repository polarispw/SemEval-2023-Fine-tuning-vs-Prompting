import os
from dataclasses import dataclass, field
from typing import Optional, List, Union


from transformers import TrainingArguments, SchedulerType, IntervalStrategy, HfArgumentParser
from transformers.trainer_utils import ShardedDDPOption
from transformers.training_args import OptimizerNames


@dataclass
class CLSDatasetArguments:
    """
    Arguments about the task dataset.
    """
    data_path: str = field(
        default="data_lib/chatgpt_review/chatgpt_reviews.csv",
        metadata={"help": "Path to the dataset"}
    )
    rand_seed: int = field(
        default=42,
        metadata={"help": "Random seed for data split."}
    )
    label2id: str = field(
        default="{'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}",
        metadata={"help": "A mapping from label to id."}
    )
    feature2input: str = field(
        default="{'input_ids': ['title', 'review'], 'label': 'rating'}",
        metadata={"help": "A mapping from feature name to input name."}
    )


@dataclass
class CLSModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="roberta-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default="model_cache/",
        metadata={"help": "Where do you want to store the pretrained models downloaded from HF"}
    )
    problem_type: Optional[str] = field(
        default=None,
        metadata={"help": "Type of the problem: `[single, multi]_label_classification`, `regression`"}
    )
    num_labels: Optional[int] = field(
        default=20,
        metadata={"help": "Number of labels to use in the last layer of the model."}
    )
    max_seq_length: Optional[int] = field(
        default=490,
        metadata={"help": "Max input length, if using virtual tokens, "
                          "max_seq_length = model.config.max_position_embeddings - num_virtual_tokens"}
    )

    # For peft
    peft_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The model id of peft part. If None then will initialize new parameters."}
    )
    peft_task_type: Optional[str] = field(
        default="SEQ_CLS",
        metadata={"help": "Task type for prompt tuning: `SEQ_CLS`, `SEQ2SEQ`"}
    )
    prompt_tuning_init_text: Optional[str] = field(
        default="Classify if the text is a complaint or not:",
        metadata={"help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"}
    )
    num_virtual_tokens: Optional[int] = field(
        default=20,
        metadata={"help": "Number of virtual tokens for prompt tuning."}
    )
    encoder_hidden_size: Optional[int] = field(
        default=128,
        metadata={"help": "Hidden size of the encoder for prompt tuning."}
    )
    ptuning_encoder_type: Optional[str] = field(
        default="MLP",
        metadata={"help": "The type of reparameterization to use for prompt tuning."}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The rank of the low-rank approximation."}
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": "The alpha value for LoRA."}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout rate for LoRA."}
    )


@dataclass
class CLSTrainingArguments(TrainingArguments):
    """
    Inherit from TrainingArguments, contains arguments usually used in CLS tasks.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.
    """

    framework = "pt"
    use_cpu = False
    lm_type: Union[str] = field(
        default="causual_lm",
        metadata={"help": "Type of the language model: `causal_lm`, `encoder`, `seq2seq`"}
    )
    task_type: Optional[str] = field(
        default="fine-tune",
        metadata={"help": "Type of the task: `fine-tune`, `pre-train`, `p-tuning`, 'prompt-tuning', 'prefix-tuning',"
                          " 'p-tuningv2', 'lora'"}
    )
    output_dir: str = field(
        default=os.path.join("archive/"),
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory. Use this to continue training if output_dir "
                          "points to a checkpoint directory."},
    )

    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."}
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run eval on the dev set."}
    )
    do_predict: bool = field(
        default=False,
        metadata={"help": "Whether to run predictions on the test set."}
    )

    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_steps: int = field(
        default=200,
        metadata={"help": "Run evaluation every X steps."},
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for AdamW."}
    )
    lr_layer_decay_rate: float = field(
        default=0.97,
        metadata={"help": "The learning rate decay rate for each layer."}
    )
    num_train_epochs: float = field(
        default=1.0,
        metadata={"help": "Total number of training epochs to perform."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Linear warmup over warmup_steps."}
    )
    freeze_encoder_layers: Optional[int] = field(
        default=-1,
        metadata={"help": "The number of layers to freeze in the base model."
                          "If -1, whole base model will be updated, if 0, the embedding layer will be frozen, etc."
                          "If > 0, the first n layers will be frozen n = min(n, encoder_layers)."}
    )

    log_file: Optional[str] = field(

        default="log.txt",
        metadata={"help": "The log file to record training process."}
    )
    logging_dir: Optional[str] = field(
        default=os.path.join("archive/tensorboard_logs/"),
        metadata={"help": "Tensorboard log dir."}
    )
    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_steps: float = field(
        default=100,
        metadata={"help": "Log every X updates steps. Should be an integer or a float in range `[0,1)` "
                          "If smaller than 1, will be interpreted as ratio of total training steps."},
    )
    save_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: float = field(
        default=200,
        metadata={"help": "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`."
                          "If smaller than 1, will be interpreted as ratio of total training steps."},
    )
    save_total_limit: Optional[int] = field(
        default=1,
        metadata={
            "help": "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                    " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                    " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                    "for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will "
                    "always be retained alongside the best model. When `save_total_limit=1` and "
                    "`load_best_model_at_end=True`,it is possible that two checkpoints are saved: the last one and "
                    "the best one (if they are different). Default is unlimited checkpoints"},
    )

    remove_unused_columns: Optional[bool] = field(
        default=True,
        metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training. When "
                          "this option is enabled, the best checkpoint will always be saved. See `save_total_limit` "
                          "for more."},
    )

    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."})

    local_rank: int = field(
        default=-1,
        metadata={"help": "For distributed training: local_rank"}
    )
    ddp_backend: Optional[str] = field(
        default=None,
        metadata={"help": "The backend to be used for distributed training",
                  "choices": ["nccl", "gloo", "mpi", "ccl"]},
    )
    sharded_ddp: Optional[Union[List[ShardedDDPOption], str]] = field(
        default="",
        metadata={
            "help": (
                "Whether or not to use sharded DDP training (in distributed training only). The base option should be"
                " `simple`, `zero_dp_2` or `zero_dp_3` and you can add CPU-offload to `zero_dp_2` or `zero_dp_3` like"
                " this: zero_dp_2 offload` or `zero_dp_3 offload`. You can add auto-wrap to `zero_dp_2` or `zero_dp_3`"
                " with the same syntax: zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`."
            ),
        },
    )
    # Do not touch this type annotation, or it will stop working in CLI
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) "
                          "or an already loaded json file as a dict"},
    )
    label_smoothing_factor: float = field(
        default=0.0,
        metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."},
    )

    optimizer: Union[OptimizerNames, str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )

    dataloader_pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether or not to pin memory for DataLoader."}
    )

    include_inputs_for_metrics: bool = field(
        default=False,
        metadata={"help": "Whether or not the inputs will be passed to the `compute_metrics` function."}
    )


if __name__ == "__main__":
    model_args = CLSModelArguments(model_name_or_path="bert-base-uncased")
    training_args = CLSTrainingArguments(task_name="llm", output_dir="../archive")

    parser = HfArgumentParser((CLSModelArguments, CLSTrainingArguments))
    # returns tuple of dataclasses
    args = parser.parse_args_into_dataclasses()
    print(args)
