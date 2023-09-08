import torch
import sys
import os

sys.path.append('../')  
from Classes.tokenizer import Tokenizer as T

train_text = open('./data/tinystoriesv2-gpt4-train.txt').read()
print(f'full training set read')
train_size = len(train_text)
chunk_size = len(train_text) // 50
numslices = train_size // chunk_size

t = T()
tokenized_text = []

for i in range(numslices):

    chunk = train_text[chunk_size * i: chunk_size * i + chunk_size]
    enc_chunk = t.encode(chunk, False, False)
    tokenized_text.extend(enc_chunk)   
    input_tns = torch.tensor(tokenized_text, dtype=torch.long)
    torch.save(input_tns, f'./data/tokenized inputs/tns_chunk_{i}.pt')
    tokenized_text = []

    if i % 5 == 0:
        print(f'step {i}/100 completed')

folder_path = './data/tokenized inputs'
files = os.listdir(folder_path)

data = []

# Loop through each file and load the tensor
for file in files:
    if file.endswith('.pt'): 
        tensor_path = os.path.join(folder_path, file)
        tensor = torch.load(tensor_path)
        data.append(tensor)        