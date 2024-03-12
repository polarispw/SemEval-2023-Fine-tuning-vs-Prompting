import json

import torch
from datasets import load_dataset, DatasetDict
from openprompt import PromptDataLoader, PromptForGeneration
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import PrefixTuningTemplate, SoftTemplate
from tqdm import tqdm
from transformers import AdamW, GPT2LMHeadModel

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
                                     tgt_text="yes" if data['label'] == 1 else "no",
                                     meta={"Premise": data['Premise'], "Conclusion": data['Conclusion'],
                                           "Stance": data['Stance'], "Description": desc_text})
        example_dataset[split].append(input_example)
print(example_dataset['train'][0])

# load the plm
model_name = "gpt2"
model_path = "../.cache/models--gpt2-medium/snapshots/f65d4965d1221eff2bcf34f53a2ba12120e18f24"
plm, tokenizer, model_config, WrapperClass = load_plm(model_name, model_path)

# Constructing Template
template_text = ('{"soft": "Question: Given the premise: "} {"meta": "Premise"} {"soft": "and the conclusion: "} '
                 '{"meta": "Conclusion"} {"meta": "Stance"} '
                 '{"soft": "it. Is it based on the value"} {"meta": "Description"} {"soft": "?"} {"mask"}')
mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text=template_text)
wrapped_example = mytemplate.wrap_one_example(example_dataset['train'][0])
print(wrapped_example)

# We provide a `PromptDataLoader` class to help you wrap them into an `torch.DataLoader` style iterator.
train_dataloader = PromptDataLoader(dataset=example_dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=5,
                                    batch_size=8, shuffle=True, teacher_forcing=True, predict_eos_token=True,
                                    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=example_dataset["validation"], template=mytemplate,
                                         tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=256,
                                         decoder_max_length=256,
                                         batch_size=8, shuffle=False, teacher_forcing=False, predict_eos_token=True,
                                         truncate_method="head")

prompt_model = PromptForGeneration(plm=plm, template=mytemplate, freeze_plm=True, tokenizer=tokenizer)
use_cuda = True
if use_cuda:
    prompt_model = prompt_model.cuda()
else:
    prompt_model = prompt_model.cpu()

# Now the training is standard
loss_func = torch.nn.CrossEntropyLoss()

# Using different optimizer for prompt parameters and model parameters
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{
    "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
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
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if step % 200 == 1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)

# Evaluate
generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "bad_words_ids": [[628], [198]]
}

alllabels = []
allpreds = []
prompt_model.eval()

for step, inputs in enumerate(tqdm(validation_dataloader)):
    if use_cuda:
        inputs = inputs.cuda()
    _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
    alllabels.extend(inputs['tgt_text'].tolist())
    allpreds.extend(output_sentence.tolist())

acc = sum([1 if j in i else 0 for i, j in zip(allpreds, alllabels)]) / len(allpreds)
print(acc)

# calculate F1 score
from sklearn.metrics import f1_score

print(f1_score(alllabels, allpreds, average='macro'))
