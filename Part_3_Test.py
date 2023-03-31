# Based on your sentiment analysis results 
# (especially looking at posts/tweets with negative and positive sentiment, 
#  as posts/tweets with neutral sentiment are less informative), 
# you will need to identify factors/reasons/topics that drive sentiment. 

# Those are factors (reasons, topics) that explain sentiment and can be used 
# for decision making and recommendations in Part 4. 

# You can use any Natural Language Processing models for this part of the project 
# (use existing models or develop your own).



import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model     = AutoModelForSequenceClassification.from_pretrained("Souvikcmsa/SentimentAnalysisDistillBERT",)
tokenizer = AutoTokenizer.from_pretrained("Souvikcmsa/SentimentAnalysisDistillBERT",)

inputs = tokenizer(["I love AutoTrain", "I am happy that you called me, but I don't like the way you talk. " ], return_tensors = "pt", padding = True)
outputs = model(**inputs)
logits = outputs["logits"].cpu().detach().numpy()
print()

for i in logits:
    print(math.exp(i[0]) / ( math.exp(i[0])+math.exp(i[1])+math.exp(i[2]) ))
    print(math.exp(i[1]) / ( math.exp(i[0])+math.exp(i[1])+math.exp(i[2]) ))
    print(math.exp(i[2]) / ( math.exp(i[0])+math.exp(i[1])+math.exp(i[2]) ))






aa=bb




from transformers import DistilBertTokenizer, DistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')


# output = model(**encoded_input)
# print(output.last_hidden_state.size())





# from transformers import GPT2LMHeadModel

# model = GPT2LMHeadModel.from_pretrained('gpt2')  # or any other checkpoint
# word_embeddings = model.transformer.wte.weight  # Word Token Embeddings 
# position_embeddings = model.transformer.wpe.weight  # Word Position Embeddings 


import torch
from transformers import GPT2Model, GPT2Tokenizer

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Tokenize the input text
input_text = "This is an example sentence."
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Get the output from the GPT-2 model
output = model(input_ids)

# Extract text embeddings
text_embeddings = output.last_hidden_state

# Generate position embeddings
position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
position_embeddings = model.wpe(position_ids)



print("Text embeddings shape:", text_embeddings.shape)


print("Position embeddings shape:", position_embeddings.shape)







from huggingface_hub import notebook_login

notebook_login()

import transformers

print(transformers.__version__)


GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "cola"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16


from datasets import load_dataset, load_metric

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

dataset



dataset["train"][0]

import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))


show_random_elements(dataset["train"])

metric

import numpy as np

fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
metric.compute(predictions=fake_preds, references=fake_labels)



from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

tokenizer("Hello, this one sentence!", "And this sentence goes with it.")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")



def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)



preprocess_function(dataset['train'][:5])

encoded_dataset = dataset.map(preprocess_function, batched=True)











