import json

import torch
from datasets import load_dataset, DatasetDict
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import SoftTemplate
from tqdm import tqdm
from transformers import AdamW, T5ForConditionalGeneration

train_dataset = load_dataset('json', data_files='../data_lib/flatten_train_set.json')['train']
validation_dataset = load_dataset('json', data_files='../data_lib/flatten_valid_set.json')['train']
with open("../data_lib/value-categories.json", "r") as f:
    description = json.load(f)

raw_dataset = DatasetDict({'train': train_dataset, 'validation': validation_dataset})
print(raw_dataset)

select_category = 'Self-direction: thought'
dataset = raw_dataset.filter(lambda example: example['label_category'] == select_category)
desc_text = [k + ": " + v[0] for k, v in description[select_category].items()]
desc_text = " ".join(desc_text)

example_dataset = {}
for split in ['train', 'validation']:
    example_dataset[split] = []
    for data in dataset[split]:
        input_example = InputExample(guid=data['Argument ID'] + data['label_category'],
                                     label=data['label'],
                                     meta={"Premise": data['Premise'], "Conclusion": data['Conclusion'],
                                           "Stance": data['Stance'], "Description": desc_text})
        example_dataset[split].append(input_example)
print(example_dataset['train'][0])

# load the plm
model_name = "roberta"
model_path = ".cache/models--roberta-large/snapshots/716877d372b884cad6d419d828bac6c85b3b18d9"
plm, tokenizer, model_config, WrapperClass = load_plm(model_name, model_path)
# plm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large",
#                                                  cache_dir="../.cache",
#                                                  device_map="auto",
#                                                  torch_dtype=torch.float16)
# if running DeBERTa, uncomment the following line
# model_config = DebertaConfig.from_pretrained(model_path)
# plm = DebertaForMaskedLM.from_pretrained(model_path, config=model_config)
# tokenizer = DebertaTokenizer.from_pretrained(model_path)
# WrapperClass = MLMTokenizerWrapper


# Constructing Template
template_text = ('{"soft": "Question: Given the premise: "} {"meta": "Premise"} {"soft": "and the conclusion: "} '
                 '{"meta": "Conclusion"} {"meta": "Stance"} '
                 '{"soft": "it. Is it based on the value"} {"meta": "Description"} {"soft": "?"} {"mask"}')
mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text=template_text)
wrapped_example = mytemplate.wrap_one_example(example_dataset['train'][0])
print(wrapped_example)

# We provide a `PromptDataLoader` class to help you wrap them into an `torch.DataLoader` style iterator.
train_dataloader = PromptDataLoader(dataset=example_dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                    batch_size=8, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="head")

# Define the verbalizer for classification
myverbalizer = ManualVerbalizer(tokenizer,
                                num_classes=2,
                                label_words=[["yes"], ["no"]])

print(myverbalizer.label_words_ids)
logits = torch.randn(2, len(tokenizer))  # creating a pseudo output from the plm, and see what the verbalizer does
print(myverbalizer.process_logits(logits))

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=True)
if use_cuda:
    prompt_model = prompt_model.cuda()

# Now the training is standard
loss_func = torch.nn.CrossEntropyLoss()

# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]
trainable_pram = 0
for n, p in prompt_model.template.named_parameters():
    if "raw_embedding" not in n:
        trainable_pram += p.numel()
print("trainable parameters: {}".format(trainable_pram))

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3)

for epoch in range(1):
    tot_loss = 0
    for step, inputs in enumerate(tqdm(train_dataloader)):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if step % 500 == 1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)

# Evaluate
validation_dataloader = PromptDataLoader(dataset=example_dataset["validation"], template=mytemplate,
                                         tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                         batch_size=4, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                         truncate_method="head")

allpreds = []
alllabels = []
for step, inputs in enumerate(tqdm(validation_dataloader)):
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
print(acc)

# calculate F1 score
from sklearn.metrics import f1_score

print(f1_score(alllabels, allpreds, average='macro'))
