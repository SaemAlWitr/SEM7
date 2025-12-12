import os
import re
import math
import time
import random
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# === USER: edit paths ====
# -------------------------
TRAIN_CSV_PATH = "/kaggle/input/col774/train_6x6_mazes.csv"   # <- set this
TEST_CSV_PATH  = "/kaggle/input/col774/test_6x6_mazes.csv"    # <- set this
MODEL_PATH     = "/kaggle/input/transformer/pytorch/default/1/transformer_epoch20.pt"  # <- set this
# -------------------------

# Device selection: prefer MPS on Mac, else GPU, else CPU
if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print("Using device:", DEVICE)

# Fixed constants from your RNN code
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
def coord_to_idx(coord: Tuple[int, int]) -> int:
    r, c = coord
    return r * 6 + c + MX

INPUT_PAD_LEN = 189
OUTPUT_PAD_LEN = 38
PAD_IDX = 0  # same padding index used in training

# -------------------------
# Parsing helpers & plotting (your functions adapted)
# -------------------------
def parse_list_string(s: str):
    return eval(s)

def parse_coords(s):
    nums = re.findall(r"-?\d+", s)
    return tuple(map(int, nums)) if len(nums) == 2 else None

def extract_between(tag, text):
    patterns = [
        rf"<\s*{tag}\s*[_\-\s]?\s*START\s*>(.*?)<\s*{tag}\s*[_\-\s]?\s*END\s*>",
        rf"<\s*{tag}START\s*>(.*?)<\s*{tag}END\s*>",
        rf"<\s*{tag}\s*START\s*>(.*?)<\s*{tag}\s*END\s*>",
        rf"<\s*{tag.replace(' ', '_')}\s*START\s*>(.*?)<\s*{tag.replace(' ', '_')}\s*END\s*>",
    ]
    for p in patterns:
        m = re.search(p, text, re.S | re.I)
        if m:
            return m.group(1).strip()
    raise ValueError(f"Could not find section for tag '{tag}'. Tried multiple patterns.")

def plot_maze(tokens):
    text = " ".join(tokens)
    adj_section = extract_between("ADJLIST", text)
    origin_section = extract_between("ORIGIN", text)
    target_section = extract_between("TARGET", text)
    path_section = extract_between("PATH", text)

    origin = parse_coords(origin_section)
    target = parse_coords(target_section)

    # parse edges like "(r,c) <--> (r2,c2)"
    edge_matches = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*<-->\s*\(\s*-?\d+\s*,\s*-?\d+\s*\)", adj_section)
    edges = []
    for em in edge_matches:
        coords = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", em)
        a = parse_coords(coords[0])
        b = parse_coords(coords[1])
        edges.append((a, b))

    # parse path coordinates (supports parenthesized coords)
    path = [parse_coords(p) for p in re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", path_section)]
    if not path:
        nums = re.findall(r"-?\d+\s*,\s*-?\d+", path_section)
        path = [tuple(map(int, re.findall(r"-?\d+", s))) for s in nums]

    if not edges:
        raise ValueError("No edges found in adjacency list. Ensure format '(r,c) <--> (r2,c2)'.")

    rows = 6
    cols = 6

    vertical_walls = np.ones((rows, cols + 1), dtype=bool)
    horizontal_walls = np.ones((rows + 1, cols), dtype=bool)

    for (r1, c1), (r2, c2) in edges:
        if r1 == r2:
            c_between = min(c1, c2) + 1
            vertical_walls[r1, c_between] = False
        elif c1 == c2:
            r_between = min(r1, r2) + 1
            horizontal_walls[r_between, c1] = False
        else:
            print(f"Warning: non-grid edge {(r1,c1)} <--> {(r2,c2)} ignored")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')

    # base light grid
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c, c + 1
            y_top = rows - r
            y_bot = rows - r - 1
            ax.plot([x0, x1], [y_top, y_top], color='lightgray', lw=2)
            ax.plot([x0, x1], [y_bot, y_bot], color='lightgray', lw=2)
            ax.plot([x0, x0], [y_bot, y_top], color='lightgray', lw=2)
            ax.plot([x1, x1], [y_bot, y_top], color='lightgray', lw=2)

    # draw walls
    for r in range(rows):
        for c in range(cols + 1):
            if vertical_walls[r, c]:
                x = c
                y_top = rows - r
                y_bot = rows - r - 1
                ax.plot([x, x], [y_bot, y_top], color='black', lw=5, solid_capstyle='butt')

    for r in range(rows + 1):
        for c in range(cols):
            if horizontal_walls[r, c]:
                y = rows - r
                ax.plot([c, c + 1], [y, y], color='black', lw=5, solid_capstyle='butt')

    if path:
        for (r, c) in path:
            x0, x1 = c, c + 1
            y_top = rows - r
            y_bot = rows - r - 1
            rect = plt.Rectangle((x0, y_bot), 1, 1, facecolor=(1, 0.9, 0.9), edgecolor=None, zorder=0)
            ax.add_patch(rect)

        path_x = [c + 0.5 for (r, c) in path]
        path_y = [rows - r - 0.5 for (r, c) in path]
        ox, oy = origin[1] + 0.5, rows - origin[0] - 0.5
        tx, ty = target[1] + 0.5, rows - target[0] - 0.5
        ax.plot(path_x, path_y, linestyle='--', linewidth=2, color='red', zorder=4)
        ax.scatter(ox, oy, c='red', s=80, marker='o', zorder=5)
        ax.scatter(tx, ty, c='red', s=80, marker='x', zorder=5)
    else:
        ox, oy = origin[1] + 0.5, rows - origin[0] - 0.5
        tx, ty = target[1] + 0.5, rows - target[0] - 0.5
        ax.scatter(ox, oy, c='red', s=80, marker='o', zorder=5)
        ax.scatter(tx, ty, c='red', s=80, marker='x', zorder=5)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    plt.yticks([])
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    plt.tight_layout()
    plt.show()

# -------------------------
# Dataset (RNN-style encoding)
# -------------------------
class MazeDatasetRNNStyle(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = []
        for i in range(df.shape[0]):
            in_tokens = parse_list_string(df.iloc[i]['input_sequence'])
            out_tokens = parse_list_string(df.iloc[i]['output_path'])

            y_in = []
            for token in in_tokens:
                if token == ';':
                    continue
                if len(token) == 5:
                    coord = eval(token)
                    y_in.append(coord_to_idx(coord))
                else:
                    y_in.append(BASE_DICT[token])

            y_out = [1]   # same as RNN code
            for token in out_tokens:
                if len(token) == 5:
                    p = eval(token)
                    y_out.append(p[0]*6 + p[1] + 3)
                else:
                    y_out.append(2)

            if len(y_in) < INPUT_PAD_LEN:
                y_in.extend([0] * (INPUT_PAD_LEN - len(y_in)))
            else:
                y_in = y_in[:INPUT_PAD_LEN]
            if len(y_out) < OUTPUT_PAD_LEN:
                y_out.extend([0] * (OUTPUT_PAD_LEN - len(y_out)))
            else:
                y_out = y_out[:OUTPUT_PAD_LEN]

            self.data.append((torch.LongTensor(y_in), torch.LongTensor(y_out)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# -------------------------
# Transformer model def (must match training architecture)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

def generate_square_subsequent_mask(sz: int):
    mask = torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1)
    float_mask = torch.zeros((sz, sz), dtype=torch.float32)
    float_mask = float_mask.masked_fill(mask, float("-inf"))
    return float_mask.to(DEVICE)

def make_padding_mask(seq: torch.Tensor, pad_idx: int):
    return (seq == pad_idx)

class TransformerMaze(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 512, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.generator = nn.Linear(d_model, vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_key_padding_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb.transpose(0,1), src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        out = self.decoder(tgt_emb.transpose(0,1), memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        out = out.transpose(0,1)
        logits = self.generator(out)
        return logits

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        logits = self.decode(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return logits

# -------------------------
# Utilities: greedy decode, metrics
# -------------------------
def greedy_decode_batch(model, src_batch, max_len=OUTPUT_PAD_LEN):
    model.eval()
    src = src_batch.to(DEVICE)
    src_key_padding_mask = make_padding_mask(src, PAD_IDX).to(DEVICE)
    with torch.no_grad():
        memory = model.encode(src, src_key_padding_mask=src_key_padding_mask)
        batch_size = src.size(0)
        ys = torch.full((batch_size, 1), 1, dtype=torch.long, device=DEVICE)  # start token = 1 for outputs
        finished = [False] * batch_size
        for _ in range(max_len - 1):
            tgt_mask = generate_square_subsequent_mask(ys.size(1))
            tgt_key_padding_mask = make_padding_mask(ys, PAD_IDX).to(DEVICE)
            logits = model.decode(ys, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
            next_logits = logits[:, -1, :]
            next_tokens = next_logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tokens], dim=1)
            for b in range(batch_size):
                if not finished[b] and next_tokens[b].item() == 2:
                    finished[b] = True
            if all(finished):
                break
        # convert ys -> python lists skipping initial 1 and stop at 2
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

def compute_token_f1(preds_list, golds_list):
    tp = fp = fn = 0
    for p, g in zip(preds_list, golds_list):
        pc = Counter(p)
        gc = Counter(g)
        common = pc & gc
        tp += sum(common.values())
        fp += sum((pc - common).values())
        fn += sum((gc - common).values())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

# -------------------------
# Load CSVs and dataset
# -------------------------
df_train = pd.read_csv(TRAIN_CSV_PATH)
df_test  = pd.read_csv(TEST_CSV_PATH)

from sklearn.model_selection import train_test_split
df_tr, df_val = train_test_split(df_train, train_size=0.9, random_state=42, stratify=df_train['maze_type'])

train_ds = MazeDatasetRNNStyle(df_tr.reset_index(drop=True))
val_ds   = MazeDatasetRNNStyle(df_val.reset_index(drop=True))
test_ds  = MazeDatasetRNNStyle(df_test.reset_index(drop=True))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

# -------------------------
# Load saved model (supports two checkpoint formats)
# -------------------------
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

if "model_state_dict" in ckpt:
    state = ckpt["model_state_dict"]
else:
    state = ckpt  # assume full state dict saved directly

if "vocab_V" in ckpt:
    vocab_size = int(ckpt["vocab_V"])
elif "vocab" in ckpt:
    vocab_size = len(ckpt["vocab"])
else:
    # fallback: derive vocab size from state keys (generator weight shape)
    # generator weight is often named 'generator.weight' or similar - search shape
    found = False
    for k, v in state.items():
        if k.endswith("generator.weight"):
            vocab_size = v.shape[0]
            found = True
            break
    if not found:
        # last resort: set to RNN V (derived from coord_to_idx((5,5)) + 1)
        vocab_size = coord_to_idx((5, 5)) + 1

print("Vocab size determined:", vocab_size)

model = TransformerMaze(vocab_size=vocab_size, d_model=128, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1, pad_idx=PAD_IDX)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# -------------------------
# Evaluate train & test seq acc + token F1
# -------------------------
def evaluate_loader(loader):
    total = 0
    exact = 0
    all_preds = []
    all_golds = []
    for src, tgt in loader:
        preds = greedy_decode_batch(model, src)
        batch_size = tgt.size(0)
        for i in range(batch_size):
            gold_seq = []
            for idx in tgt[i].tolist()[1:]:
                if idx == 2:
                    break
                if idx != 0 and idx != 1:
                    gold_seq.append(idx)
            pred_seq = preds[i]
            all_preds.append(pred_seq)
            all_golds.append(gold_seq)
            if pred_seq == gold_seq:
                exact += 1
            total += 1
    seq_acc = exact / total if total > 0 else 0.0
    prec, rec, f1 = compute_token_f1(all_preds, all_golds)
    return seq_acc, prec, rec, f1

# print("Evaluating on TRAIN (this may take some time)...")
# train_seq_acc, train_prec, train_rec, train_f1 = evaluate_loader(train_loader)
# print(f"Train: seq acc = {train_seq_acc:.4f}, token F1 = {train_f1:.4f} (P={train_prec:.4f}, R={train_rec:.4f})")

# print("Evaluating on TEST ...")
# test_seq_acc, test_prec, test_rec, test_f1 = evaluate_loader(test_loader)
# print(f"Test: seq acc = {test_seq_acc:.4f}, token F1 = {test_f1:.4f} (P={test_prec:.4f}, R={test_rec:.4f})")

# -------------------------
# Visualize 5 random examples from train & test
# -------------------------
def idxs_to_coords_tokens(idxs: List[int]) -> List[str]:
    coords = []
    for x in idxs:
        if x >= 3 and x <= 38:
            val = x - 3
            r = val // 6
            c = val % 6
            coords.append(f"({r},{c})")
        elif x == 1:
            coords.append("<PATH_START>")
        elif x == 2:
            coords.append("<PATH_END>")
        else:
            break
    return coords

def visualize_random_examples(df_source, dataset, loader, sample_count=5):
    n = len(dataset)
    indices = random.sample(range(n), sample_count)
    for idx in indices:
        # original tokens from CSV
        input_tokens = parse_list_string(df_source.iloc[idx]['input_sequence'])
        gold_path_tokens = parse_list_string(df_source.iloc[idx]['output_path'])
        # get dataset encoded src to feed model
        src_tensor, tgt_tensor = dataset[idx]
        preds = greedy_decode_batch(model, src_tensor.unsqueeze(0), max_len=OUTPUT_PAD_LEN)
        pred_idxs = preds[0]
        pred_tokens_for_plot = idxs_to_coords_tokens(pred_idxs)
        if (pred_tokens_for_plot[-1] != '<PATH_END>'):
            pred_tokens_for_plot.append('<PATH_END>')
        print("Example index:", idx)
        print("Gold path tokens:", gold_path_tokens)
        print("Predicted path token indices:", pred_idxs)
        print("Predicted path tokens:", pred_tokens_for_plot)
        # show gold
        print("Visualizing GOLD path:")
        plot_maze(input_tokens + gold_path_tokens)
        # show prediction
        print("Visualizing PREDICTED path:")
        plot_maze(input_tokens + pred_tokens_for_plot)

# Visualize 5 from train (use df_tr)
print("Visualizing 5 random TRAIN examples (gold vs predicted):")
visualize_random_examples(df_tr.reset_index(drop=True), train_ds, train_loader, sample_count=5)

print("Visualizing 5 random TEST examples (gold vs predicted):")
visualize_random_examples(df_test.reset_index(drop=True), test_ds, test_loader, sample_count=5)

# Done
print("Evaluation + visualization complete.")