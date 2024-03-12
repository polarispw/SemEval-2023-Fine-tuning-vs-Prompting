"""
This file contains a dict that maps task name to their corresponding methods
To customize a new task, add a new item to this dict and implement the methods
"""
from peft import (
    PromptTuningConfig,
    PromptEncoderConfig,
    PrefixTuningConfig,
    LoraConfig, PromptTuningInit, PeftConfig,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from data.data_manager import CLSDataManager
from data.metric import compute_acc_f1, compute_simcse
from models.modeling_bert import SIMBert
from train.bert_trainer import BertTrainer


class TaskMethodMap:
    def __init__(self, model_args, data_args, training_args):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.task_type = training_args.task_type

        self.task_methods_dic = {
            "fine-tune": {
                "dataset": CLSDataManager,
                "metric": compute_acc_f1,
                "data_collator": DataCollatorWithPadding,
                "tokenizer": AutoTokenizer,
                "model": AutoModelForSequenceClassification,
                "trainer": BertTrainer,
            },

            "pre-train": {
                "dataset": CLSDataManager,
                "metric": compute_simcse,
                "data_collator": DataCollatorWithPadding,
                "tokenizer": AutoTokenizer,
                "model": SIMBert,
                "trainer": BertTrainer,
            },

            # not support yet
            "prompt-tuning": {
                "dataset": CLSDataManager,
                "metric": None,
                "data_collator": DataCollatorWithPadding,
                "tokenizer": AutoTokenizer,
                "model": AutoModelForSequenceClassification,
                "peft_config": PromptTuningConfig,
                "trainer": Trainer,
            },

            "p-tuning": {
                "dataset": CLSDataManager,
                "metric": compute_acc_f1,
                "data_collator": DataCollatorWithPadding,
                "tokenizer": AutoTokenizer,
                "model": AutoModelForSequenceClassification,
                "peft_config": PromptEncoderConfig,
                "trainer": Trainer,
            },

            "prefix-tuning": {
                "dataset": CLSDataManager,
                "metric": compute_acc_f1,
                "data_collator": DataCollatorWithPadding,
                "tokenizer": AutoTokenizer,
                "model": AutoModelForSequenceClassification,
                "peft_config": PrefixTuningConfig,
                "trainer": Trainer,
            },

            "p-tuningv2": {
                "dataset": CLSDataManager,
                "metric": compute_acc_f1,
                "data_collator": DataCollatorWithPadding,
                "tokenizer": AutoTokenizer,
                "model": AutoModelForSequenceClassification,
                "peft_config": PrefixTuningConfig,
                "trainer": Trainer,
            },

            # not support yet
            "lora": {
                "dataset": CLSDataManager,
                "metric": None,
                "data_collator": DataCollatorWithPadding,
                "tokenizer": AutoTokenizer,
                "model": AutoModelForSequenceClassification,
                "peft_config": LoraConfig,
                "trainer": Trainer,
            },
        }

    def get_datamanager(self, tokenizer):
        return self.task_methods_dic[self.task_type]['dataset'](self.data_args, tokenizer)

    def get_data_collator(self, tokenizer):
        return self.task_methods_dic[self.task_type]['data_collator'](tokenizer=tokenizer)

    def get_metric(self):
        return self.task_methods_dic[self.task_type]['metric']

    def get_tokenizer(self):
        if any(k in self.model_args.model_name_or_path for k in ("gpt", "opt", "bloom")):
            padding_side = "left"
        else:
            padding_side = "right"

        return self.task_methods_dic[self.task_type]['tokenizer'].from_pretrained(self.model_args.model_name_or_path,
                                                                                  model_max_length=self.model_args.max_seq_length,
                                                                                  padding_side=padding_side,
                                                                                  cache_dir=self.model_args.cache_dir)

    def get_model(self):
        if self.task_type == 'pre-train':
            return self.task_methods_dic[self.task_type]['model'](self.model_args)
        elif self.task_type == 'fine-tune':
            return self.task_methods_dic[self.task_type]['model'].from_pretrained(
                self.model_args.model_name_or_path,
                num_labels=self.model_args.num_labels,
                cache_dir=self.model_args.cache_dir,
                ignore_mismatched_sizes=True
            )
        else:
            return self.task_methods_dic[self.task_type]['model'].from_pretrained(
                self.model_args.model_name_or_path,
                num_labels=self.model_args.num_labels,
                cache_dir=self.model_args.cache_dir,
                return_dict=True,
                ignore_mismatched_sizes=True
            )

    def get_peft_config(self):
        if self.model_args.peft_model_id is not None:
            print(f"Loading peft config from {self.model_args.peft_model_id}")
            return PeftConfig.from_pretrained(self.model_args.peft_model_id)

        if self.task_type == 'prompt-tuning':
            return self.task_methods_dic[self.task_type]['peft_config'](
                task_type=self.model_args.peft_task_type,
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=self.model_args.num_virtual_tokens,
                prompt_tuning_init_text=self.model_args.prompt_tuning_init_text,
                tokenizer_name_or_path=self.model_args.model_name_or_path,
            )
        elif self.task_type == 'p-tuning':
            return self.task_methods_dic[self.task_type]['peft_config'](
                task_type=self.model_args.peft_task_type,
                num_virtual_tokens=self.model_args.num_virtual_tokens,
                encoder_hidden_size=self.model_args.encoder_hidden_size,
                encoder_reparameterization_type=self.model_args.ptuning_encoder_type,
            )
        elif self.task_type == 'prefix-tuning':
            return self.task_methods_dic[self.task_type]['peft_config'](
                task_type=self.model_args.peft_task_type,
                num_virtual_tokens=self.model_args.num_virtual_tokens,
                prefix_projection=True,
            )
        elif self.task_type == 'p-tuningv2':
            return self.task_methods_dic[self.task_type]['peft_config'](
                task_type=self.model_args.peft_task_type,
                num_virtual_tokens=self.model_args.num_virtual_tokens,
                prefix_projection=False,
            )
        elif self.task_type == 'lora':
            return self.task_methods_dic[self.task_type]['peft_config'](
                task_type=self.model_args.peft_task_type,
                inference_mode=self.training_args.do_train,
                r=self.model_args.lora_rank,
                lora_alpha=self.model_args.lora_alpha,

                lora_dropout=self.model_args.lora_dropout,
            )
        else:
            raise NotImplementedError
