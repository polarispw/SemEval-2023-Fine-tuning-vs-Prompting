import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime

from datasets import load_dataset
from filelock import FileLock
from peft import get_peft_model
from transformers import HfArgumentParser

from config.args_list import CLSModelArguments, CLSDatasetArguments, CLSTrainingArguments
from models.modeling_bert import CLSBert
from utils.task_methods_map import TaskMethodMap

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():

    # Parse arguments
    parser = HfArgumentParser((CLSModelArguments, CLSDatasetArguments, CLSTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    now = datetime.now()
    dt_str = now.strftime('%m_%d_%H_%M_%S')
    task_type = training_args.task_type
    suffix = f"{task_type}_{dt_str}"
    training_args.output_dir = os.path.join(training_args.output_dir, suffix)
    training_args.logging_dir = os.path.join(training_args.logging_dir, training_args.output_dir)

    # Setup logging
    if os.path.isfile(training_args.log_file):
        os.remove(training_args.log_file)
    logging.basicConfig(
        filename="tmp_log.txt",
        filemode='a',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    # Create task methods map
    task_methods_map = TaskMethodMap(model_args, data_args, training_args)

    # Create tokenizer
    tokenizer = task_methods_map.get_tokenizer()
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    data_manager = task_methods_map.get_datamanager(tokenizer)
    train_dataset = load_dataset('json', data_files='data_lib/merged_train_set.json')['train']
    valid_dataset = load_dataset('json', data_files='data_lib/merged_valid_set.json')['train']
    train_dataset = data_manager.collate_for_model(raw_ds=train_dataset,
                                                   feature2input={"input_text": ["Conclusion", "Stance", "Premise"]})

    valid_dataset = data_manager.collate_for_model(raw_ds=valid_dataset,
                                                   feature2input={"input_text": ["Conclusion", "Stance", "Premise"]})
    print("\n", train_dataset.features)

    # Create data collator
    data_collator = task_methods_map.get_data_collator(tokenizer)

    # Load metric
    metric = task_methods_map.get_metric()

    # Create model
    model = CLSBert(model_args)
    model.to(training_args.device)
    model.classifiers.to(training_args.device)

    # Initialize Trainer
    trainer = task_methods_map.task_methods_dic[task_type]['trainer'](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metric,
    )

    # Training
    train_result = None
    if training_args.do_train:
        train_result = trainer.train()

    # Evaluation
    eval_result = None
    if training_args.do_eval:
        eval_result = trainer.evaluate()

    final_result = {
        'time': str(datetime.today()),
        'output_dir': training_args.output_dir,
        'train_result': train_result,
        'eval_result': eval_result,
    }

    # Save args and logs
    with open(os.path.join(training_args.output_dir, 'run_config.json'), 'w') as j:
        # merge args into one dict
        args_dict = asdict(model_args)
        args_dict.update(asdict(data_args))
        args_dict.update(asdict(training_args))
        json.dump(args_dict, j, indent=4)

    log_file = os.path.join(training_args.output_dir, training_args.log_file)
    if trainer.is_world_process_zero():
        with FileLock('log.lock'):
            with open(log_file, 'a') as f:
                final_result = json.dumps(final_result, indent=4)
                f.write(str(final_result) + '\n')
                with open("tmp_log.txt", 'r') as tmp:
                    f.write(tmp.read())

    logger.info('****** Output Dir *******')
    logger.info(training_args.output_dir)

    return final_result


if __name__ == "__main__":
    main()
    logging.shutdown()
    os.remove("tmp_log.txt")

    # visualize the training logs by tensorboard
    # tensorboard --logdir archive/tensorboard_logs
