"""
This is a general trainer for CLS task
It inherits from the transformers Trainer class
"""

from torch.optim import SGD
from transformers import Trainer
from transformers.optimization import AdamW


class BertTrainer(Trainer):
    """
    Customize optimizer to enable tricks like layer-wise lr decay, and freezing layers
    """

    def create_optimizer(self):
        """
        Based on Transformers' default one, we add fixing layer option and layer wise learning rate.
        It works well on BERT-like encoder models. For seq2seq or decoder models, it needs to be overwritten.
        """
        # No existing optimizer, create the optimizer
        if self.optimizer is None:

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = []

            # layer wise lr: calculate the lr_factor for each layer
            num_layer = 12
            base_factor = 1.0
            lr_factor_list = []
            for i in range(num_layer):
                lr_factor_list.append(base_factor)
                base_factor *= self.args.lr_layer_decay_rate
            lr_factor_list = lr_factor_list[::-1]

            # group the parameters
            for n, p in self.model.named_parameters():
                if "embeddings" in n:
                    if self.args.freeze_encoder_layers == -1:
                        optimizer_grouped_parameters.append(
                            {
                                "params": [p],
                                "weight_decay": 0.0,
                                "lr": self.args.learning_rate * lr_factor_list[0]
                            }
                        )
                    continue
                elif "base.encoder.layer" in n:
                    layer_num = n[n.find('encoder.layer') + 14:].split('.')[0]
                    layer_num = int(layer_num)
                    weight_norm = not any(nd in n for nd in no_decay)
                    if layer_num >= self.args.freeze_encoder_layers:
                        optimizer_grouped_parameters.append(
                            {
                                "params": [p],
                                "weight_decay": self.args.weight_decay if weight_norm else 0.0,
                                "lr": self.args.learning_rate * lr_factor_list[layer_num],
                            }
                        )
                        # print(n, self.args.learning_rate * lr_factor_list[layer_num])
                    continue
                else:
                    # this part is the classifier
                    optimizer_grouped_parameters.append(
                        {
                            "params": [p],
                            "weight_decay": self.args.weight_decay if 'weight' in n else 0.0,
                            "lr": self.args.learning_rate,
                        }
                    )

            # create the optimizer
            if self.args.optimizer == 'adamw_torch':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == 'sgd':
                self.optimizer = SGD(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError

        else:
            # the model already has an optimizer
            return


if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding
    import numpy as np
    import evaluate

    from config.args_list import CLSTrainingArguments, CLSModelArguments
    from models.modeling_bert import SIMBert

    raw_datasets = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized_datasets = raw_datasets.map(lambda example:
                                          tokenizer(example["sentence1"], example["sentence2"], truncation=True),
                                          batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model_args = CLSModelArguments(model_name_or_path='bert-base-uncased', cache_dir='../model_cache')
    my_model = SIMBert(model_args)


    def compute_metrics(eval_preds):
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = CLSTrainingArguments(output_dir="../archive",
                                         evaluation_strategy="epoch",
                                         max_steps=500,
                                         learning_rate=2e-5,
                                         logging_steps=10,
                                         save_steps=500,
                                         lr_layer_decay_rate=0.95,
                                         freeze_encoder_layers=10,
                                         )

    trainer = BertTrainer(
        my_model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # visualize the training logs by tensorboard
    # tensorboard --logdir .archive/tensorboard_logs
