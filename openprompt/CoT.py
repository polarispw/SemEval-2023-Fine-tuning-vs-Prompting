import sys
import argparse

import pandas as pd
from datasets import Dataset
import torch
import evaluate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)

def transform_text_to_labels(results):
    ans = []
    for text in results:
        if "No" in text[:-4]:
            ans.append(0)
        elif "Yes" in text[:-4]:
            ans.append(1)
        else:
            ans.append(-1)
    return ans

def add_new_column(example, ValueDescription):
    Conclusion = example["Conclusion"]
    Stance = example["Stance"]
    Premise = example["Premise"]
    example["text"] = f"<s>[INST] Instruction: Only use Yes or No to answer the quesiton\nQuestion: Given the premise: {Premise} and the conclusion:\n{Conclusion} {Stance} it. Is it based on the {ValueDescription}?\nThe answer is [/INST]"
    return example

def get_model(tokenizer_path, model_path):
    use_4bit = True
    bnb_4bit_quant_type = "nf4"
    bnb_4bit_compute_dtype = "float16"
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    use_nested_quant = False
    device_map = "auto"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map)
    model.config.use_cache = False

    return tokenizer, model

def transform_text_to_labels(results):
    ans = []
    for result in results:
      text = result[0]['generated_text']
      if "No" in text[:-4]:
          ans.append(0)
      elif "Yes" in text[:-4]:
          ans.append(1)
      else:
          ans.append(-1)
    return ans

def compute_acc_f1(pred, labels):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    acc = accuracy.compute(predictions=pred, references=labels)['accuracy']
    f1_score = f1.compute(predictions=pred, references=labels, average='macro')['f1']
    return acc, f1_score

print("hello")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str)   
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--label_path', type=str)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path
    model_path = args.model_path
    test_data_path = args.data_path
    test_label_path = args.label_path

    prompts = "cot"
    # tasks = ["Self-direction: thought"]
    tasks = ['Argument ID', 'Self-direction: thought', 'Self-direction: action',
       'Stimulation', 'Hedonism', 'Achievement', 'Power: dominance',
       'Power: resources', 'Face', 'Security: personal', 'Security: societal',
       'Tradition', 'Conformity: rules', 'Conformity: interpersonal',
       'Humility', 'Benevolence: caring', 'Benevolence: dependability',
       'Universalism: concern', 'Universalism: nature',
       'Universalism: tolerance', 'Universalism: objectivity']
    max_length = 200

    df = pd.read_csv(test_data_path, sep='\t')
    dataset = Dataset.from_pandas(df)
    label = pd.read_csv(test_label_path, sep='\t')
    
    tokenizer, model = get_model(model_path)
    text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length)

    for task in tasks:
        tmp_dataset = dataset.map(
            lambda example, ValueDescription=task: add_new_column(example, ValueDescription)
        )
    
        output = text_gen(tmp_dataset[:]['text'])
        pred = transform_text_to_labels(output)
        acc, f1 = compute_acc_f1(pred, label[task].to_list())
        print(f"tasks:{task}, accuracy is {acc}, f1 is {f1}")
