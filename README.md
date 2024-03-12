# Classify all in one

## Features

- support various models: encoders(BERT, ALBERT, RoBERTa), decoders(T5), seq2seq(GPT2)
- support functions like full parameter training, fine-tuning, peft
- customize the structure of model, invisible to higher levels like data processor or trainer
- easily run with config.json and CLI

## Project structure

```python
Project/
|---configs/
    |---run_config.json	# hyper-parameters to run.py
    |---args_list.py	# args dataclass
|---data/
    |---augmentation.py	# data augmentaion
    |---file_utils.py	# utils for collate_data_files.py
    |---data_manager.py	# dataset class
    |---data_collator.py
    |---metric.py
|---data_lib/	# dir of datasets, manually created
|---models/
    |---modeling_bert.py
    |---...
|---train/
    |---trainer.py
    |---bert_trainer.py
    |---...
|---utils/
    |---chatgpt_api.py	# access to chatgpt
    |---task_methods_map.py
run.py
requirements.txt
```

## Usage

> the project based on 🤗[huggingface's](https://huggingface.co/docs) repo: `transformers`, `datasets`, `peft`, `evaluate`

### Quick start

1. prepare your env by `requirements.txt`
2. copy your dataset to `data_lib`, and implement your data manager. You can refer to the glue_mrpc dataset in `test.py` for your first try since it is well prepared by huggingface.
3. check the args in `config/run_config.json`, and change them to fit your task
4. use following command to start training:

   ```shell
   python run.py .\config\run_config.json
   ```

   records will be saved in `./archive`

### Customize

- We use a `TaskMethodMap` Class to manage methods to be called when training models.
- To add your own models, just check the `task_methods_dic` and add your own methods. They can be either API from huggingface or coded by yourself.

  ```python
  "[METHOD_NAME]": {
      "dataset": [DATA_MANAGER],
      "metric": [METRIC_FUNC],
      "data_collator": [DATA_COLLATOR],
      "tokenizer": [TOKENIZER],
      "model": [MODEL],
      "peft_config": [PEFT_CONFIG],
      "trainer": [TRAINER],
  }
  ```
