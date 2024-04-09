from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification

import numpy
import pandas as pd
import torch

df = pd.read_csv('train.csv',  encoding = "ISO-8859-1")

text = df['text']

print(text.describe())

print("number of null values" , text.isnull().sum())

text = text.dropna()

print("text shape == ", text.shape)

text = text.head(100)

text = text.tolist()

raw_inputs = text

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# # # Tokenize inputs
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print("inputs ", inputs)

model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

print(outputs.logits.shape)

print(outputs.logits)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print("predictions " , predictions)

print(model.config.id2label)

labels = torch.argmax(predictions, dim=-1)

labels = torch.where(predictions[:, 0] > predictions[:, 1], torch.tensor(0), torch.tensor(1))

labels_array = labels.numpy()

df_labels = pd.DataFrame(labels_array, columns=['Labels'])

df_labels['Labels'] = df_labels['Labels'].apply(lambda x: "Positive" if x == 1 else "Negative")

text  = pd.DataFrame(text, columns=['Text'])

final_df = pd.concat([text, df_labels], axis = 1)

print("final_df ", final_df)
