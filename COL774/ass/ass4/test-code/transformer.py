import os
import math
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TRAIN_CSV = "/kaggle/input/col774/train_6x6_mazes.csv"
TEST_CSV = "/kaggle/input/col774/test_6x6_mazes.csv"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
D_MODEL = 128  
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

INPUT_PAD_LEN = 189
OUTPUT_PAD_LEN = 38

PAD_IDX = 0

BASE_DICT = {
    '<ADJLIST_START>': 1,
    '<ADJLIST_END>': 2,
    '<ORIGIN_START>': 3,
    '<ORIGIN_END>': 4,
    '<TARGET_START>': 5,
    '<TARGET_END>': 6,
    '<PATH_START>': 7,
    '<-->': 8
}
MX = max(BASE_DICT.values()) + 1 
def coord_to_idx(coord):
    r, c = coord
    return r * 6 + c + MX
V = coord_to_idx((5, 5)) + 1

print("MX:", MX, "Vocab size V:", V)

class MazeDatasetRNNStyle(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = []
        for i in range(df.shape[0]):
            in_tokens = eval(df.iloc[i]['input_sequence'])
            out_tokens = eval(df.iloc[i]['output_path'])
            y_in = []
            for token in in_tokens:
                if token == ';':
                    continue
                if len(token) == 5:
                    coord = eval(token)
                    y_in.append(coord_to_idx(coord))
                else:
                    y_in.append(BASE_DICT[token])

            y_out = [1] 
            for token in out_tokens:
                if len(token) == 5:
                    p = eval(token)
                    y_out.append(p[0] * 6 + p[1] + 3)
                else:
                    y_out.append(2)
            if len(y_in) < INPUT_PAD_LEN:
                y_in.extend([0] * (INPUT_PAD_LEN - len(y_in)))
            else:
                y_in = y_in[:INPUT_PAD_LEN]
            y_out.extend([0] * (OUTPUT_PAD_LEN - len(y_out)))
            
            self.data.append((torch.LongTensor(y_in), torch.LongTensor(y_out)))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

from sklearn.model_selection import train_test_split
df_tr, df_val = train_test_split(df_train, train_size=0.9, random_state=SEED, stratify=df_train['maze_type'])

train_dataset = MazeDatasetRNNStyle(df_tr.reset_index(drop=True))
val_dataset = MazeDatasetRNNStyle(df_val.reset_index(drop=True))
test_dataset = MazeDatasetRNNStyle(df_test.reset_index(drop=True))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Dataset sizes -> train:", len(train_dataset), "val:", len(val_dataset), "test:", len(test_dataset))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

def sq_subseq_mask_gen(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1).to(DEVICE)
    float_mask = torch.zeros((sz, sz), device=DEVICE, dtype=torch.float32)
    float_mask = float_mask.masked_fill(mask, float("-inf"))
    return float_mask

class TransformerMaze(nn.Module):
    def __init__(self, vocab_size, d_model = D_MODEL, nhead = NHEAD,
                 num_layers = NUM_LAYERS, dim_feedforward = DIM_FEEDFORWARD,
                 dropout = DROPOUT, pad_idx = PAD_IDX):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation='relu')
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.generator = nn.Linear(d_model, vocab_size)

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, scr_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb.transpose(0, 1), scr_mask=scr_mask)
        return memory

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_mask=None, mem_pad=None):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        out = self.decoder(tgt_emb.transpose(0, 1), memory,
                           tgt_mask=tgt_mask,
                           tgt_key_mask=tgt_key_mask,
                           mem_pad=mem_pad)
        out = out.transpose(0, 1)
        logits = self.generator(out)
        return logits

    def forward(self, src, tgt, scr_mask=None, tgt_mask=None, tgt_key_mask=None, mem_pad=None):
        memory = self.encode(src, scr_mask=scr_mask)
        logits = self.decode(tgt, memory, tgt_mask=tgt_mask, tgt_key_mask=tgt_key_mask, mem_pad=mem_pad)
        return logits

from collections import Counter

def compute_token_f1(preds_list, golds_list):
    tp = 0
    fp = 0
    fn = 0
    for p, g in zip(preds_list, golds_list):
        pc = Counter(p)
        gc = Counter(g)
        common = pc & gc
        tp += sum(common.values())
        fp += sum((pc - common).values())
        fn += sum((gc - common).values())
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1

def greedy_decode(model, src, max_len = OUTPUT_PAD_LEN):
    model.eval()
    src = src.to(DEVICE)
    scr_mask = (src == PAD_IDX).to(DEVICE)
    with torch.no_grad():
        memory = model.encode(src, scr_mask=scr_mask)
        batch_size = src.size(0)
        ys = torch.full((batch_size, 1), fill_value=1, dtype=torch.long, device=DEVICE)
        finished = [False] * batch_size
        for _ in range(max_len - 1):
            tgt_mask = sq_subseq_mask_gen(ys.size(1))
            tgt_key_mask = (ys == PAD_IDX).to(DEVICE)
            logits = model.decode(ys, memory, tgt_mask=tgt_mask, tgt_key_mask=tgt_key_mask, mem_pad=scr_mask)
            next_logits = logits[:, -1, :]
            next_tokens = next_logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tokens], dim=1)
            for i in range(batch_size):
                if not finished[i] and next_tokens[i].item() == 2:
                    finished[i] = True
            if all(finished):
                break
        results = []
        for i in range(batch_size):
            seq = []
            for idx in ys[i].tolist()[1:]:
                if idx == 2:
                    break
                if idx != 0:
                    seq.append(idx)
            results.append(seq)
    return results

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for src, tgt in dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        scr_mask = (src == PAD_IDX).to(DEVICE)
        tgt_key_mask = (tgt_input == PAD_IDX).to(DEVICE)
        tgt_mask = sq_subseq_mask_gen(tgt_input.size(1))

        optimizer.zero_grad()
        logits = model(src, tgt_input, scr_mask=scr_mask, tgt_mask=tgt_mask, tgt_key_mask=tgt_key_mask, mem_pad=scr_mask)
        vocab_size = logits.size(-1)
        loss = criterion(logits.view(-1, vocab_size), tgt_output.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        num_tokens = (tgt_output != PAD_IDX).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    avg_loss = total_loss /total_tokens
    return avg_loss

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    exact = 0
    total = 0
    all_preds = []
    all_golds = []
    with torch.no_grad():
        for src, tgt in dataloader:
            tgt = tgt.to(DEVICE)
            src = src.to(DEVICE)
            tgt_output = tgt[:, 1:]
            tgt_input = tgt[:, :-1]

            scr_mask = (src == PAD_IDX).to(DEVICE)
            tgt_key_mask = (tgt_input == PAD_IDX).to(DEVICE)
            tgt_mask = sq_subseq_mask_gen(tgt_input.size(1))

            logits = model(src, tgt_input, scr_mask=scr_mask, tgt_mask=tgt_mask, tgt_key_mask=tgt_key_mask, mem_pad=scr_mask)
            vocab_size = logits.size(-1)
            loss = criterion(logits.view(-1, vocab_size), tgt_output.contiguous().view(-1))

            preds = logits.argmax(dim=-1)
            for i in range(preds.size(0)):
                pred_seq = []
                gold_seq = []
                for idx in preds[i].tolist():
                    if idx == 2:
                        break
                    if idx != 0 and idx != 1:
                        pred_seq.append(idx)
                for idx in tgt[i].tolist()[1:]:
                    if idx == 2:
                        break
                    if idx != 0 and idx != 1:
                        gold_seq.append(idx)
                all_preds.append(pred_seq)
                all_golds.append(gold_seq)
                if pred_seq == gold_seq:
                    exact += 1
                total += 1

            num_tokens = (tgt_output != PAD_IDX).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    seq_acc = exact / total
    prec, rec, f1 = compute_token_f1(all_preds, all_golds)
    return avg_loss, seq_acc, prec, rec, f1

model = TransformerMaze(V, D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, pad_idx=PAD_IDX)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses = []
val_seq_accs = []
val_token_f1s = []

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_seq_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion)
    end = time.time()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_seq_accs.append(val_seq_acc)
    val_token_f1s.append(val_f1)

    print(f"Epoch {epoch}/{EPOCHS} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val seq acc: {val_seq_acc:.4f} | Val token F1: {val_f1:.4f} | Time: {end-start:.1f}s")

    ckpt_dir = "saved_transformer"
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "vocab_V": V
    }, os.path.join(ckpt_dir, f"transformer_epoch{epoch}.pt"))

plt.figure(figsize=(8,4))
plt.plot(range(1, EPOCHS+1), train_losses, label='train_loss')
plt.plot(range(1, EPOCHS+1), val_losses, label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(range(1, EPOCHS+1), val_seq_accs, label='val_seq_acc')
plt.plot(range(1, EPOCHS+1), val_token_f1s, label='val_token_f1')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.grid(True)
plt.show()

val_iter = iter(val_loader)
src_batch, tgt_batch = next(val_iter)
preds = greedy_decode(model, src_batch[:8].to(DEVICE), max_len=OUTPUT_PAD_LEN)
print("First 8 predictions (token indices lists):")
for p in preds:
    print(p)
print("First 8 gold outputs (token indices lists):")
for i in range(8):
    gold = tgt_batch[i].tolist()[1:]
    gold_seq = []
    for idx in gold:
        if idx == 2:
            break
        if idx != 0 and idx != 1:
            gold_seq.append(idx)
    print(gold_seq)

