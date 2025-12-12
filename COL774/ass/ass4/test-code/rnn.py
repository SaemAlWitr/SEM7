import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import os
df = pd.read_csv('train_6x6_mazes.csv')
df_test = pd.read_csv('test_6x6_mazes.csv')
df.head()

def parse_coords(s):
    nums = re.findall(r"-?\d+", s)
    return tuple(map(int, nums)) if len(nums) == 2 else None

def extract_between(tag, text):
    """Accepts many tag styles: <TAG_START>, <TAG START>, <TAG-START>, <TAGSTART>, etc."""
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
        # fallback: "r,c" tokens without parentheses
        nums = re.findall(r"-?\d+\s*,\s*-?\d+", path_section)
        path = [tuple(map(int, re.findall(r"-?\d+", s))) for s in nums]

    if not edges:
        raise ValueError("No edges found in adjacency list. Ensure format '(r,c) <--> (r2,c2)'.")

    # --------------------------
    # Grid size (cells indexed with (0,0) = top-left)
    # --------------------------
    all_nodes = {n for e in edges for n in e if n is not None}
    all_nodes.update([origin, target])
    all_nodes.update([p for p in path if p is not None])
    rows = 6
    cols = 6


    vertical_walls = np.ones((rows, cols + 1), dtype=bool)
    horizontal_walls = np.ones((rows + 1, cols), dtype=bool)

    for (r1, c1), (r2, c2) in edges:
        if r1 == r2:
            # same row, adjacent columns -> remove vertical wall between them
            c_between = min(c1, c2) + 1  # column index of the vertical segment between c and c+1
            vertical_walls[r1, c_between] = False
        elif c1 == c2:
            # same column, adjacent rows -> remove horizontal wall between them
            r_between = min(r1, r2) + 1  # row index of horizontal segment between r and r+1
            horizontal_walls[r_between, c1] = False
        else:
            # diagonal or invalid — ignore, but warn
            print(f"Warning: non-grid edge {(r1,c1)} <--> {(r2,c2)} ignored")


    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')

    # Draw a full light-gray grid (every cell border)
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c, c + 1
            y_top = rows - r
            y_bot = rows - r - 1
            ax.plot([x0, x1], [y_top, y_top], color='lightgray', lw=2)   # top
            ax.plot([x0, x1], [y_bot, y_bot], color='lightgray', lw=2)   # bottom
            ax.plot([x0, x0], [y_bot, y_top], color='lightgray', lw=2)   # left
            ax.plot([x1, x1], [y_bot, y_top], color='lightgray', lw=2)   # right

    # Draw vertical walls (black) using vertical_walls[r,c]
    for r in range(rows):
        for c in range(cols + 1):
            if vertical_walls[r, c]:
                x = c
                y_top = rows - r
                y_bot = rows - r - 1
                ax.plot([x, x], [y_bot, y_top], color='black', lw=5, solid_capstyle='butt')

    # Draw horizontal walls (black)
    for r in range(rows + 1):
        for c in range(cols):
            if horizontal_walls[r, c]:
                y = rows - r
                ax.plot([c, c + 1], [y, y], color='black', lw=5, solid_capstyle='butt')

    shade_path_cells = True
    if shade_path_cells and path:
        for (r, c) in path:
            # rectangle corners in plot coords
            x0, x1 = c, c + 1
            y_top = rows - r
            y_bot = rows - r - 1
            rect = plt.Rectangle((x0, y_bot), 1, 1, facecolor=(1, 0.9, 0.9), edgecolor=None, zorder=0)
            ax.add_patch(rect)

    # Plot path line and markers (convert (r,c) top-left -> matplotlib coords)
    print(origin, target)
    if path:
        path_x = [c + 0.5 for (r, c) in path]
        path_y = [rows - r - 0.5 for (r, c) in path]
        ax.plot(path_x, path_y, linestyle='--', linewidth=2, color='red', zorder=4)
        ax.scatter(origin[1]+.5, rows-origin[0]-.5, c='red', s=80, marker='o', zorder=5)  # start
        ax.scatter(target[1]+.5, rows-target[0]-.5, c='red', s=80, marker='x', zorder=5)  # goal
    else:
        # if no path, still mark origin/target
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


device = torch.device(0)


class BahdanauAttention(nn.Module):
    def __init__(self, n_enc, n_dec, n_attn):
        super().__init__()
        self.W_a = nn.Linear(n_dec, n_attn, bias=False)
        self.U_a = nn.Linear(n_enc, n_attn, bias=False)
        self.v_a = nn.Linear(n_attn, 1, bias=False)

    def forward(self, h, s_i_1,mask):
        """
        h:(batch, seq_len, n)
        s_{i-1}: (batch, n)  -> prev hidden state of decoder
        """
        energy = torch.tanh(self.U_a(h) + self.W_a(s_i_1).unsqueeze(1))

        alpha = self.v_a(energy).squeeze(-1)
        if mask is not None:
            alpha = alpha.masked_fill(~mask, -1e4)
        attn_weights = F.softmax(alpha, dim=-1)

        # context: (batch, hidden_size)
        context = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)

        return context, attn_weights

class RNNenc(nn.Module):
    def __init__(self, K_x, m_x, n_enc, padding_idx=0, dropout=0):
        super().__init__()
        self.embeder = nn.Embedding(K_x, m_x, padding_idx=padding_idx)
        self.rnn = nn.RNN(
            input_size=m_x,
            hidden_size=n_enc,
            num_layers=2,
            nonlinearity='tanh',
            batch_first=True,
            dropout = dropout
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        emb = self.dropout(self.embeder(x))
        h, h_last = self.rnn(emb)
        return h, h_last

class RNNdec(nn.Module):
    def __init__(self, K_y, m_y, n_dec, n_enc, n_attn,out_dim, paddin_idx,dropout=0):
        super().__init__()
        self.embeder = nn.Embedding(K_y, m_y,padding_idx=paddin_idx)
        self.attention = BahdanauAttention(n_enc,n_dec=n_dec, n_attn=n_attn)
        self.rnn = nn.RNN(m_y+n_enc, 
                          n_dec,
                          num_layers=2,
                          batch_first=True,
                          nonlinearity='tanh',dropout=dropout)
        self.out = nn.Linear(n_dec+m_y+n_attn, out_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self, h, h_last, y=None, T_y=None, 
                teacher_forcing = 0.5, mask=None,starter=1):
        batch_size = h.size(0)
        teacher_size = int(batch_size*teacher_forcing)
        device = h.device
        if y is not None: T_y = y.size(1)-1
        out_len = T_y
        s_t = h_last
        y_in = torch.full((batch_size,),starter, device=device, dtype=torch.long)
        outputs = []
        all_attn_weights = []
        for t in range(out_len):
            emb = self.dropout(self.embeder(y_in)).unsqueeze(1)
            context, attn_weights = self.attention.forward(h, s_t[-1], mask=mask)
            context_unsq = context.unsqueeze(1) # so that rnn runs for 1 unit
            rnn_in = torch.cat([emb, context_unsq], dim=2)
            all_attn_weights.append(attn_weights.unsqueeze(1)) # unsqueeze so we can cat all later on
            _, s_t = self.rnn(rnn_in, s_t)
            out_in = torch.cat([s_t[-1],context,emb.squeeze(1)], dim=1)
            logits = self.out(out_in) # (batch,out_dim)
            outputs.append(logits.unsqueeze(1))
            if t < out_len-1:
                y_in = logits.argmax(dim=1)
                if y is not None:
                    # teacher forcing
                    idxs = np.arange(batch_size)
                    np.random.shuffle(idxs)
                    y_in[idxs[:teacher_size]] = y[idxs[:teacher_size],t+1]
        outputs = torch.cat(outputs, dim=1)
        all_attn_weights = torch.cat(all_attn_weights,dim=1)
        return outputs, all_attn_weights

class seq2seqBahdanau(nn.Module):
    def __init__(self, K_x,m_x, K_y,m_y, n_enc, n_attn, n_dec, out_dim,dropout=0):
        super().__init__()
        self.encoder = RNNenc(K_x, m_x, n_enc, padding_idx=0, dropout=dropout)
        self.decoder = RNNdec(K_y, m_y, n_dec, n_enc, n_attn, out_dim, paddin_idx=0,dropout=dropout)
    def forward(self, X, y=None, T_y=None, teacher_forcing=0.5, mask=None, starter = 1):
        h, h_last = self.encoder(X)
        logits, attn_weights = self.decoder(h, h_last, y, T_y, teacher_forcing, mask, starter)
        return logits, attn_weights

def collate_fn(batch, pad_idx=0):
    inp = [item[0].unsqueeze(0) for item in batch]
    out = [item[1].unsqueeze(0) for item in batch]
    inp_seqs = torch.cat(inp, dim = 0)  # (batch, T_x)
    out_paths = torch.cat(out, dim = 0) # (batch, T_y)
    inp_mask = (inp_seqs != pad_idx)
    return {'inp': inp_seqs, 'out': out_paths, 'inp_mask': inp_mask}


from torch.amp import autocast, GradScaler
def train_epoch(model:seq2seqBahdanau, dataloader: torch.utils.data.DataLoader, optimizer, criterion, device, clip=5, teacher_forcing=0.5, scaler=None):
    model.train()
    total_loss = 0
    for batch in (dataloader):
        inp = batch['inp'].to(device)
        out = batch['out'].to(device)
        inp_mask = batch['inp_mask'].to(device)
        optimizer.zero_grad()
        with autocast(device.type):
            outputs, _ = model(inp,out, T_y=None,teacher_forcing=teacher_forcing, mask=inp_mask, starter=1)
            loss = criterion(outputs.view(-1, outputs.size(-1)), out[:,1:].contiguous().view(-1))
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        total_loss += loss.item()*inp.size(0)
    return total_loss/len(dataloader.dataset)


def f(x):
    if x==0:
        return '0'
    if x==1:
        return '<PATH_START>'
    if x == 2:
        return '<PATH_END>'
    return f"({(x-3)//6},{(x-3)%6})"
def decode(preds: torch.Tensor, eos = 2, f=None, device=torch.device(0)):
    T_x = preds.size(1)
    eos_idx = (preds == eos).int().argmax(dim = 1)+1
    for i in range(eos_idx.size(0)):
        if preds[i,eos_idx[i]-1]!=2:
            eos_idx[i] = T_x
            preds[i,-1]=2
    mask = torch.tensor([[True]*eos_idx[i].item()+[False]*(T_x-eos_idx[i].item()) for i in range(preds.size(0))], dtype=torch.bool, device=device)
    preds_pad = torch.multiply(preds, mask)
    if f is not None:
        seqs = [[f(j.item()) for j in preds_pad[i,:eos_idx[i].item()]] for i in range(preds.size(0))]
    else: seqs=None
    return preds_pad, seqs

from torch.utils.data import DataLoader
from time import time
from sklearn.metrics import *
from collections import Counter

def evaluate(model, dataloader, device=torch.device(0), pad_idx=0, eos = 2):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_seq_wrongs = 0
    total_tp = total_fp = total_fn = 0

    with torch.no_grad():
        for batch in dataloader:
            inp = batch['inp'].to(device)
            out = batch['out'].to(device)
            inp_mask = batch['inp_mask'].to(device)

            logits, _ = model(inp, out, teacher_forcing=0, mask=inp_mask) # logits: (B, T_y-1, V)
            preds = logits.argmax(dim=2) # (B, T_y)
            preds, _ = decode(preds)
            targets = out[:, 1:].contiguous() # (B, T_y-1)
            # mask to ignore PAD tokens in targets
            tgt_mask = (targets != pad_idx)
            num_tokens = int(tgt_mask.sum().item())
            if num_tokens == 0:
                continue

            V = logits.size(-1)
            logits_flat = logits.view(-1, V) # (B*T_y-1, V)
            targets_flat = targets.view(-1) # (B*T_y-1,)

            per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # (B*T_y-1,)
            per_token_loss = per_token_loss.view(targets.size()) # (B, T_y-1)
            token_loss_sum = (per_token_loss * tgt_mask).sum().item()

            total_loss += token_loss_sum
            total_tokens += num_tokens
            total_seq_wrongs += int(((preds != targets) & tgt_mask).any(dim=1).sum().item())

            # micro F1 counts (exclude pads)
            for p_row, t_row, m_row in zip(preds, targets, tgt_mask):
                # filter out pad positions
                if not m_row.any():
                    continue
                p_list = p_row[m_row].tolist()
                t_list = t_row[m_row].tolist()
                pred_ctr = Counter(p_list)
                act_ctr = Counter(t_list)
                tp = sum((pred_ctr & act_ctr).values())
                fp = sum((pred_ctr - act_ctr).values())
                fn = sum((act_ctr - pred_ctr).values())
                total_tp += tp
                total_fp += fp
                total_fn += fn

    # Final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    seq_accuracy = 1.0 - (total_seq_wrongs / len(dataloader.dataset))

    if (total_tp + total_fp) > 0:
        p = total_tp / (total_tp + total_fp)
    else:
        p = 0.0
    if (total_tp + total_fn) > 0:
        r = total_tp / (total_tp + total_fn)
    else:
        r = 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    return avg_loss, seq_accuracy, f1

def fit(model:seq2seqBahdanau, train_dataset, val_dataset, test_taset, pading_idx, batch_size=32, epochs=20,lr=1e-4,
        device=None,save_path="chkpt.pt",teacher_forcing=0.5):
    device=device or torch.device(0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, pad_idx=pading_idx))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, pad_idx=pading_idx))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, pad_idx=pading_idx))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pading_idx)
    scaler = GradScaler(device.type) if torch.cuda.is_available() else None
    model.to(device)
    losses = []
    accs = []
    f1s = []
    for epoch in tqdm(range(1, epochs+1)):
        start = time()
        print(f"Starting Epoch {epoch}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 clip=5, scaler=scaler, teacher_forcing=teacher_forcing)
        print(f"Evaluating. Train loss={train_loss}")
        train_loss, train_acc, train_f1 = evaluate(model, train_loader, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, device)
        losses.append((train_loss, val_loss, test_loss))
        accs.append((train_acc, val_acc, test_acc))
        f1s.append((train_f1, val_f1, test_f1))
        end = time()
        print(f"Epoch {epoch} | Time: {end-start:.1f}s \nTrain loss={train_loss:.4f} | Val loss={val_loss:.4f} | Test loss={test_loss:.4f}")
        print(f"Train acc={train_acc:.4f} | Val acc={val_acc:.4f} | Test acc={test_acc:.4f}")
        print(f"Train F1-score={train_f1:.4f} | Val F1-score={val_f1:.4f} | Test F1-score={test_f1:.4f}")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss': val_loss
        }, str(epoch)+'_'+save_path)
        print(f"model saved to {epoch}_{save_path}")
    return {"model":model, "losses":losses, "accs": accs, "f1s":f1s}

from tqdm import tqdm

class MazeDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        d = {'<ADJLIST_START>':1,'<ADJLIST_END>':2, '<ORIGIN_START>':3,'<ORIGIN_END>':4,'<TARGET_START>':5,'<TARGET_END>':6, '<PATH_START>':7,';':8,'<-->':9}
        mx = max(d.values())+1
        f = lambda x: x[0]*6+x[1]+mx
        V = f((5,5))+1
        print(V)
        def encode(df):
            y = []
            for i in tqdm(range(df.shape[0])):
                out_tokens = eval(df.iloc[i]['output_path'])
                in_tokens = eval(df.iloc[i]['input_sequence'])
                y_out = [1]
                y_in = []
                for token in in_tokens:
                    # if token in (,): continue
                    y_in.append(f(eval(token)) if len(token) == 5 else d[token])
                for token in out_tokens:
                    if len(token) == 5:
                        p = eval(token)
                        y_out.append(p[0]*6+p[1]+3)
                    else: y_out.append(2)
                if len(y_in) < 249: y_in.extend([0]*(249-len(y_in))) #pad
                if len(y_out) < 38: y_out.extend([0]*(38-len(y_out))) 
                y.append((torch.LongTensor(y_in), torch.LongTensor(y_out))) 
            return y
        self.data = encode(df)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

print("creating datasets for the model with all the tokens and dropout 0.1")
df_train, df_val = train_test_split(df, train_size=0.9, random_state=42, stratify=df['maze_type'])
train_dataset = MazeDataset(df_train)
val_dataset = MazeDataset(df_val)
test_dataset = MazeDataset(df_test)


model = seq2seqBahdanau(46,128,39,128,512,512,512,39,dropout=0.1)
out = fit(model, train_dataset, val_dataset, test_dataset, 0, epochs=20, device=torch.device(0), save_path="checkpoint_clip_5;_--_do-10.pt")

losses, accs, f1s = out['losses'], out['accs'], out['f1s']
eps = list(range(1, len(losses)+1))
for metric, name in zip([losses, accs, f1s], ['loss', 'accuracy', 'F1-Score']):
    fig = plt.figure()
    for i, tp in enumerate(['train', 'val', 'test']):
        values = [val[i] for val in metric]
        plt.plot(eps, values, label = tp)
    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.legend()
    plt.title(f"Epoch vs {name}")
    plt.savefig(f"Epoch-vs-{name}.png", dpi=300)
    print(f"Plot saved at Epoch-vs-{name}.png")
    plt.close(fig)


d = {'losses':losses, 'accs':accs, 'f1s':f1s}
torch.save(d, "METRICS.pt")

# class MazeDataset(torch.utils.data.Dataset):
#     def __init__(self, df):
#         super().__init__()
#         d = {'<ADJLIST_START>':1,'<ADJLIST_END>':2, '<ORIGIN_START>':3,'<ORIGIN_END>':4,'<TARGET_START>':5,'<TARGET_END>':6, '<PATH_START>':7}
#         mx = max(d.values())+1
#         f = lambda x: x[0]*6+x[1]+mx
#         V = f((5,5))+1
#         print(V)
#         def encode(df):
#             y = []
#             for i in tqdm(range(df.shape[0])):
#                 out_tokens = eval(df.iloc[i]['output_path'])
#                 in_tokens = eval(df.iloc[i]['input_sequence'])
#                 y_out = [1]
#                 y_in = []
#                 for token in in_tokens:
#                     if token in (';','<-->'): continue
#                     y_in.append(f(eval(token)) if len(token) == 5 else d[token])
#                 for token in out_tokens:
#                     if len(token) == 5:
#                         p = eval(token)
#                         y_out.append(p[0]*6+p[1]+3)
#                     else: y_out.append(2)
#                 if len(y_in) < 129: y_in.extend([0]*(129-len(y_in))) #pad
#                 if len(y_out) < 38: y_out.extend([0]*(38-len(y_out))) 
#                 y.append((torch.LongTensor(y_in), torch.LongTensor(y_out))) 
#             return y
#         self.data = encode(df)
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, index):
#         return self.data[index]
# print("creating datasets for the model with all the tokens (except ; and <-->) and dropout 0.1")

# df_train, df_val = train_test_split(df, train_size=0.9, random_state=42, stratify=df['maze_type'])
# train_dataset = MazeDataset(df_train)
# val_dataset = MazeDataset(df_val)
# test_dataset = MazeDataset(df_test)


# model = seq2seqBahdanau(44,128,39,128,512,512,512,39)
# out = fit(model, train_dataset, val_dataset, test_dataset, 0, device=torch.device(0), save_path="checkpoint_clip_5.pt")


# losses, accs, f1s = out['losses'], out['accs'], out['f1s']
# eps = list(range(1, len(losses)+1))
# for metric, name in zip([losses, accs, f1s], ['loss', 'accuracy', 'F1-Score']):
#     fig = plt.figure()
#     for i, tp in enumerate(['train', 'val', 'test']):
#         values = [val[i] for val in metric]
#         plt.plot(eps, values, label = tp)
#     plt.xlabel('epoch')
#     plt.ylabel(name)
#     plt.legend()
#     plt.title(f"Epoch vs {name}")
#     plt.savefig(f"/kaggle/working/Epoch-vs-{name}.png", dpi=300)
#     print(f"Plot saved at /kaggle/working/Epoch-vs-{name}.png")
#     plt.close(fig)
# d = {'losses':losses, 'accs':accs, 'f1s':f1s}
# torch.save(d, "METRICS.pt")

# class MazeDataset(torch.utils.data.Dataset):
#     def __init__(self, df):
#         super().__init__()
#         d = {'<ADJLIST_START>':1,'<ADJLIST_END>':2, '<ORIGIN_START>':3,'<ORIGIN_END>':4,'<TARGET_START>':5,'<TARGET_END>':6, '<PATH_START>':7, ';':8}
#         mx = max(d.values())+1
#         f = lambda x: x[0]*6+x[1]+mx
#         V = f((5,5))+1
#         print(V)
#         def encode(df):
#             y = []
#             for i in tqdm(range(df.shape[0])):
#                 out_tokens = eval(df.iloc[i]['output_path'])
#                 in_tokens = eval(df.iloc[i]['input_sequence'])
#                 y_out = [1]
#                 y_in = []
#                 for token in in_tokens:
#                     if token in ('<-->',): continue
#                     y_in.append(f(eval(token)) if len(token) == 5 else d[token])
#                 for token in out_tokens:
#                     if len(token) == 5:
#                         p = eval(token)
#                         y_out.append(p[0]*6+p[1]+3)
#                     else: y_out.append(2)
#                 if len(y_in) < 129: y_in.extend([0]*(129-len(y_in))) #pad
#                 if len(y_out) < 38: y_out.extend([0]*(38-len(y_out))) 
#                 y.append((torch.LongTensor(y_in), torch.LongTensor(y_out))) 
#             return y
#         self.data = encode(df)
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, index):
#         return self.data[index]
    
# print("creating datasets for the model with all the tokens except <--> and dropout 0.1")

# df_train, df_val = train_test_split(df, train_size=0.9, random_state=42, stratify=df['maze_type'])
# train_dataset = MazeDataset(df_train)
# val_dataset = MazeDataset(df_val)
# test_dataset = MazeDataset(df_test)


# model = seq2seqBahdanau(45,128,39,128,512,512,512,39)
# out = fit(model, train_dataset, val_dataset, test_dataset, 0, device=torch.device(0), save_path="checkpoint_clip_5;.pt")


# losses, accs, f1s = out['losses'], out['accs'], out['f1s']
# eps = list(range(1, len(losses)+1))
# for metric, name in zip([losses, accs, f1s], ['loss', 'accuracy', 'F1-Score']):
#     fig = plt.figure()
#     for i, tp in enumerate(['train', 'val', 'test']):
#         values = [val[i] for val in metric]
#         plt.plot(eps, values, label = tp)
#     plt.xlabel('epoch')
#     plt.ylabel(name)
#     plt.legend()
#     plt.title(f"Epoch vs {name}")
#     plt.savefig(f"/kaggle/working/Epoch-vs-{name}.png", dpi=300)
#     print(f"Plot saved at /kaggle/working/Epoch-vs-{name}.png")
#     plt.close(fig)
# d = {'losses':losses, 'accs':accs, 'f1s':f1s}
# torch.save(d, "METRICS.pt")

# class MazeDataset(torch.utils.data.Dataset):
#     def __init__(self, df):
#         super().__init__()
#         d = {'<ADJLIST_START>':1,'<ADJLIST_END>':2, '<ORIGIN_START>':3,'<ORIGIN_END>':4,'<TARGET_START>':5,'<TARGET_END>':6, '<PATH_START>':7,'<-->':8}
#         mx = max(d.values())+1
#         f = lambda x: x[0]*6+x[1]+mx
#         V = f((5,5))+1
#         print(V)
#         def encode(df):
#             y = []
#             for i in tqdm(range(df.shape[0])):
#                 out_tokens = eval(df.iloc[i]['output_path'])
#                 in_tokens = eval(df.iloc[i]['input_sequence'])
#                 y_out = [1]
#                 y_in = []
#                 for token in in_tokens:
#                     if token in (';',): continue
#                     y_in.append(f(eval(token)) if len(token) == 5 else d[token])
#                 for token in out_tokens:
#                     if len(token) == 5:
#                         p = eval(token)
#                         y_out.append(p[0]*6+p[1]+3)
#                     else: y_out.append(2)
#                 if len(y_in) < 189: y_in.extend([0]*(189-len(y_in))) #pad
#                 if len(y_out) < 38: y_out.extend([0]*(38-len(y_out))) 
#                 y.append((torch.LongTensor(y_in), torch.LongTensor(y_out))) 
#             return y
#         self.data = encode(df)
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, index):
#         return self.data[index]

# print("creating datasets for the model with all the tokens except ; and dropout 0.1")

# df_train, df_val = train_test_split(df, train_size=0.9, random_state=42, stratify=df['maze_type'])
# train_dataset = MazeDataset(df_train)
# val_dataset = MazeDataset(df_val)
# test_dataset = MazeDataset(df_test)


# model = seq2seqBahdanau(45,128,39,128,512,512,512,39)
# out = fit(model, train_dataset, val_dataset, test_dataset, 0, device=torch.device(0), save_path="checkpoint_clip_5_--.pt")


# losses, accs, f1s = out['losses'], out['accs'], out['f1s']
# eps = list(range(1, len(losses)+1))
# for metric, name in zip([losses, accs, f1s], ['loss', 'accuracy', 'F1-Score']):
#     fig = plt.figure()
#     for i, tp in enumerate(['train', 'val', 'test']):
#         values = [val[i] for val in metric]
#         plt.plot(eps, values, label = tp)
#     plt.xlabel('epoch')
#     plt.ylabel(name)
#     plt.legend()
#     plt.title(f"Epoch vs {name}")
#     plt.savefig(f"/kaggle/working/Epoch-vs-{name}.png", dpi=300)
#     print(f"Plot saved at /kaggle/working/Epoch-vs-{name}.png")
#     plt.close(fig)
# d = {'losses':losses, 'accs':accs, 'f1s':f1s}
# torch.save(d, "METRICS.pt")


# model = seq2seqBahdanau(46,128,39,128,512,512,512,39,dropout=0.1)
# model.to(device)
# train_loader = DataLoader(train_dataset, batch_size=72000, shuffle=False,
#                               collate_fn=lambda b: collate_fn(b, pad_idx=0))
# val_loader = DataLoader(val_dataset, batch_size=8000, shuffle=False,
#                           collate_fn=lambda b: collate_fn(b, pad_idx=0))
# test_loader = DataLoader(test_dataset, batch_size=20000, shuffle=False,
#                           collate_fn=lambda b: collate_fn(b, pad_idx=0))


# d = {'<ADJLIST_START>':1,'<ADJLIST_END>':2, '<ORIGIN_START>':3,'<ORIGIN_END>':4,'<TARGET_START>':5,'<TARGET_END>':6, '<PATH_START>':7,';':8,'<-->':9}
# d_inv = {d[i]:i for i in d}


# def decode_inp(inp):
#     toks = []
#     for ipt in inp:
#         tok = []
#         for i in ipt:
#             i = i.item()
#             if i == 0: continue
#             if i > 9: tok.append(f'({(i-10)//6},{(i-10)%6})')
#             else: tok.append(d_inv[i])
#         toks.append(tok)
#     return toks


# for batch in test_loader:
#     start = torch.randint(0,len(batch['inp'])-5,(1,)).item()

#     inp = (batch['inp'][start:start+5]).to(device)
#     out = (batch['out'][start:start+5]).to(device)
#     inp_mask = (batch['inp_mask'][start:start+5]).to(device)
#     logits,_ = model(inp,out,teacher_forcing=0.0,mask=inp_mask)
#     preds = logits.argmax(dim=2)

#     _,toks = decode(preds,device=device, f=f)
#     break
    

# inp_toks = decode_inp(inp)
# path_toks = toks
# for i in range(5):
#     tokens = inp_toks[i]+path_toks[i]
#     plot_maze(tokens)





