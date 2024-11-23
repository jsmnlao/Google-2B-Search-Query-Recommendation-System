# TODOS:
#   * move the training to the GPU using .device()
#   * save the model to a file after it is trained
#   * add a flag to load an existing model or train a new one
#   * train on the full data after we know it is training on GPUs
#
# In the end you'll have three or so files:
#   1) Define the model classes
#   2) Load the model classes/definition and run training, finally saving the model weights
#   3) Load the model classes, load the model weights, accept input for seeding prediction and generate results

import nltk
import pandas as pd
import numpy as np
import tiktoken
import torch

nltk.download('punkt')

splits = {'train': 'nq_open/train-00000-of-00001.parquet', 'validation': 'nq_open/validation-00000-of-00001.parquet'}
dft = pd.read_parquet("hf://datasets/google-research-datasets/nq_open/" + splits["train"])
dfv = pd.read_parquet("hf://datasets/google-research-datasets/nq_open/" + splits["validation"])
df = pd.concat([dft, dfv])

df = df.drop(columns=['answer'])
# print(df.head(10))

enc = tiktoken.get_encoding("cl100k_base") #loading encoding

DELIMITER = "|"

# This only keeps the first 2k records to make iteration on the model
# faster. The full training data set will need to be used for real
# runs of the model.
training_blob = "|".join(df['question'].to_list()[:2000])
TRAINING_SIZE = len(training_blob)

training_blob_encoded = enc.encode(training_blob)

unique_tokens = set(training_blob_encoded)

ordnial_to_token = {i: v for i, v in enumerate(sorted(unique_tokens))}
token_to_ordinal = {v: i for i, v in enumerate(sorted(unique_tokens))}

def encode_ticktokens(ticktoken_tokens: list[int]) -> list[int]:
  return [token_to_ordinal[t] for t in ticktoken_tokens]

def decode_to_ticktokens(ordinals: list[int]) -> list[int]:
  return [ordnial_to_token[t] for t in ordinals]

training_blob_double_encoded = encode_ticktokens(training_blob_encoded)

training_data_tensor = torch.tensor(training_blob_double_encoded, dtype=torch.long)

holdout_size = int(len(training_data_tensor) * .1)
holdout_size

test_data = training_data_tensor[:holdout_size]
training_data = training_data_tensor[holdout_size:]
len(test_data), len(training_data)

BATCH_SIZE = 4
BLOCK_SIZE = 8

def get_batch(split):
  data = training_data if split == 'train' else test_data
  ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

  x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
  y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
  return x, y

# print(get_batch('train'))

VOCAB_SIZE = len(set(training_blob_double_encoded))
print(VOCAB_SIZE)

import torch
import torch.nn as nn
from torch.nn import functional as F
import random

VOCAB_SIZE = len(set(training_blob_double_encoded))
NUM_EMBEDDINGS = VOCAB_SIZE // 2

# @torch.no_grad()
# def estimate_loss():
#   out = {}
#   model.eval()
#   for split in ['train', 'val']:
#     losses = torch.zeros(eval_iters)
#     for k in range(eval_iters):
#       X, Y = get_batch(split)
#       logits, loss = model(X, Y)
#       losses[k] = loss.item()
#     out[split] = losses.mean()
#   model.train()
#   return out

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(NUM_EMBEDDINGS, head_size, bias=False)
    self.query = nn.Linear(NUM_EMBEDDINGS, head_size, bias=False)
    self.value = nn.Linear(NUM_EMBEDDINGS, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    v = self.value(x)
    out = wei @ v
    return out
  
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(NUM_EMBEDDINGS, NUM_EMBEDDINGS)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out
  
class FeedForward(nn.Module):
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embed, 4*n_embed),
        nn.ReLU(),
        nn.Linear(4*n_embed, n_embed),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embed, n_head):
    super().__init__()
    head_size = n_embed // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, NUM_EMBEDDINGS)
    self.position_embedding_table = nn.Embedding(BLOCK_SIZE, NUM_EMBEDDINGS)
    #self.sa_head = Head(NUM_EMBEDDINGS)
    self.feed_forward = FeedForward(NUM_EMBEDDINGS)
    self.sa_heads = MultiHeadAttention(4, NUM_EMBEDDINGS//4)
    self.lm_head = nn.Linear(NUM_EMBEDDINGS, vocab_size)
    self.blocks = nn.Sequential(
      Block(NUM_EMBEDDINGS, n_head=4),
      Block(NUM_EMBEDDINGS, n_head=4),
      Block(NUM_EMBEDDINGS, n_head=4),
      nn.LayerNorm(NUM_EMBEDDINGS),
    )
    self.proj = nn.Linear(NUM_EMBEDDINGS, NUM_EMBEDDINGS)

  def forward(self, token, targets=None):
    token_embeddings = self.token_embedding_table(token)
    B, T = token.shape
    position_embeddings = self.position_embedding_table(torch.arange(T))
    x = token_embeddings + position_embeddings
    x = self.blocks(x)

    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

#   def generate(self, idx, max_new_tokens):
#     for _ in range(max_new_tokens):
#       logits, _ = self(idx)
#       logits = logits[:, -1, :] # (B, C)
#       probs = F.softmax(logits, dim=-1) # (B, C)
#       idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
#       idx = torch.cat((idx, idx_next), dim=1)
#     return idx
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -BLOCK_SIZE:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :] # (B, C)
      probs = F.softmax(logits, dim=-1) # (B, C)
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

m = BigramLanguageModel(VOCAB_SIZE)
# z = torch.zeros((1,1), dtype=torch.long)
# z[0][0] = random.randint(0, VOCAB_SIZE-1) # randomly seed the first token
# print(enc.decode(decode_to_ticktokens(m.generate(z, 40)[0].tolist())))

LEARNING_RATE = 1e-3
MAX_ITERS = 1000
EVAL_INTERVAL = 100
EVAL_ITERS = 100

@torch.no_grad()
def estimate_loss():
  out = {}
  m.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
      X, Y = get_batch(split)
      logits, loss = m(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  m.train()
  return out
     
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

for iter in range(MAX_ITERS):

  if iter % EVAL_INTERVAL == 0:
    losses = estimate_loss()
    print(f"iter {iter}; train loss {losses['train']:.4f}; val loss {losses['val']:.4f}")

  xb, yb = get_batch('train')

  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print(f"Final Loss: {loss.item()}")

z = torch.zeros((1,1), dtype=torch.long)

input = "where did they "
print(input)
print(enc.encode(input))
input_double_encoded = [encode_ticktokens(enc.encode(input))]
print(input_double_encoded)

example_token_tensor = torch.tensor(input_double_encoded, dtype=torch.long)
print(example_token_tensor)

print(enc.decode(decode_to_ticktokens(m.generate(example_token_tensor, 100)[0].tolist())).split('|')[0])

#for i in range(10):
#    z[0][0] = random.randint(0, VOCAB_SIZE-1) # randomly seed the first token
#    prediction = enc.decode(decode_to_ticktokens(m.generate(z, 100)[0].tolist()))
#    prediction = prediction.split('|')[0]
#    print(f'{i}) {prediction}')


