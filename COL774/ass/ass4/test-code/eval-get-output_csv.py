import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd	
import math
class MazeDataset_RNN(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        d = {'<ADJLIST_START>':1,'<ADJLIST_END>':2, '<ORIGIN_START>':3,'<ORIGIN_END>':4,'<TARGET_START>':5,'<TARGET_END>':6, '<PATH_START>':7,';':8,'<-->':9}
        mx = max(d.values())+1
        f = lambda x: x[0]*6+x[1]+mx
        V = f((5,5))+1
        def encode(df):
            y = []
            for i in range(df.shape[0]):
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

class MazeDatasetTransformer(torch.utils.data.Dataset):
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
    def forward(self, X, y, T_y=None, teacher_forcing=0.5, mask=None, starter = 1):
        h, h_last = self.encoder(X)
        logits, attn_weights = self.decoder(h, h_last, y, T_y, teacher_forcing, mask, starter)
        return logits, attn_weights
    def get_embeddings(self, X):
        return self.encoder.embeder(X)

def collate_fn(batch, pad_idx=0):
    inp = [item[0].unsqueeze(0) for item in batch]
    out = [item[1].unsqueeze(0) for item in batch]
    inp_seqs = torch.cat(inp, dim = 0)  # (batch, T_x)
    out_paths = torch.cat(out, dim = 0) # (batch, T_y)
    inp_mask = (inp_seqs != pad_idx)
    return {'inp': inp_seqs, 'out': out_paths, 'inp_mask': inp_mask}

def f(x):
    if x==0:
        return '0'
    if x==1:
        return '<PATH_START>'
    if x == 2:
        return '<PATH_END>'
    return f"({(x-3)//6},{(x-3)%6})"

def decode(preds: torch.Tensor, eos = 2, f=f, device=torch.device(0)):
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
def coord_to_idx(coord) -> int:
    r, c = coord
    return r * 6 + c + MX

INPUT_PAD_LEN = 189
OUTPUT_PAD_LEN = 38
PAD_IDX = 0  # same padding index used in training

def coord_to_idx(coord) -> int:
    r, c = coord
    return r * 6 + c + MX

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
    

def generate_square_subsequent_mask(sz: int, device):
    mask = torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1)
    float_mask = torch.zeros((sz, sz), dtype=torch.float32)
    float_mask = float_mask.masked_fill(mask, float("-inf"))
    return float_mask.to(device)

def greedy_decode_batch(model, src_batch, max_len=OUTPUT_PAD_LEN, device = None):
    model.eval()
    src = src_batch.to(device)
    src_key_padding_mask = (src == PAD_IDX).to(device)
    with torch.no_grad():
        memory = model.encode(src, src_key_padding_mask=src_key_padding_mask)
        batch_size = src.size(0)
        ys = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)  # start token = 1 for outputs
        finished = [False] * batch_size
        for _ in range(max_len - 1):
            tgt_mask = generate_square_subsequent_mask(ys.size(1), device)
            tgt_key_padding_mask = (ys == PAD_IDX).to(device)
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

def idxs_to_coords_tokens(idxs):
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



if __name__ == '__main__':
    import sys
    import torch
    import pandas as pd

    model_path = sys.argv[1]
    model_type = sys.argv[2]
    data_path  = sys.argv[3]
    output_path = sys.argv[4]
    device = torch.device(0)
    weights = torch.load(model_path)
    DATA = pd.read_csv(data_path)
    if model_type == 'rnn':
        state_dict = torch.load(model_path)['model_state']
        dataset = MazeDataset_RNN(DATA)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        model = seq2seqBahdanau(46,128,39,128,512,512,512,39,dropout=0.1)
        model.load_state_dict(state_dict)
        model.to(device)
        outputs = []
        for batch in dataloader:
            inp = batch['inp'].to(device)
            out = batch['out'].to(device)
            inp_mask = batch['inp_mask'].to(device)
            logits, _ = model(inp, y=None, T_y=37, teacher_forcing=0.0, mask=inp_mask, starter=1)
            preds = logits.argmax(dim=2)
            _, decoded_seqs = decode(preds, eos=2, f=f, device=device)
            outputs.extend(str(i) for i in decoded_seqs)
        output_df = pd.DataFrame({'output_path': outputs})
        output_df.to_csv(output_path ,index=False)
    else:
        ckpt = torch.load(model_path)
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        if "vocab_V" in ckpt:
            vocab_size = int(ckpt["vocab_V"])
        elif "vocab" in ckpt:
            vocab_size = len(ckpt["vocab"])
        else:
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
        model.to(device)
        dataset = MazeDatasetTransformer(DATA)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        outputs = []
        for src, tgt in dataloader:
            preds = greedy_decode_batch(model, src,device=device)
            decoded_preds = [idxs_to_coords_tokens(i)+['<PATH_END>'] for i in preds]
            outputs.extend(decoded_preds)
        output_df = pd.DataFrame({'output_path': outputs})
        output_df.to_csv(output_path ,index=False)





