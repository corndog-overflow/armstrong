# monk_torch8.py (PyTorch版 多GPU分布式训练 + 生成)

import glob
import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from music21 import converter, instrument, note, chord, stream

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3, dim_feedforward=512, seq_len=30):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src)
        out = out.mean(dim=1)
        out = self.fc(out)
        return out

# Dataset
class MIDIDataset(Dataset):
    def __init__(self, tokens, note_to_int, seq_len=30):
        self.seq_len = seq_len
        self.tokens = tokens
        self.note_to_int = note_to_int
        self.data = [(tokens[i:i+seq_len], tokens[i+seq_len]) for i in range(len(tokens) - seq_len)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_in, seq_out = self.data[idx]
        seq_in = torch.tensor([self.note_to_int[n] for n in seq_in], dtype=torch.long)
        seq_out = torch.tensor(self.note_to_int[seq_out], dtype=torch.long)
        return seq_in, seq_out

# Load notes
def get_notes(directory="./jazz_and_stuff"):
    tokens = []
    for file in glob.glob(directory + "/*.mid"):
        midi = converter.parse(file)
        try:
            notes = instrument.partitionByInstrument(midi).parts[0].recurse()
        except:
            notes = midi.flat.notes

        prev_offset = -1
        for element in notes:
            duration = round(element.quarterLength * 2) / 2
            if isinstance(element, note.Note):
                if prev_offset >= 0 and element.offset - prev_offset > 0.5:
                    tokens.append(f"REST_{round((element.offset - prev_offset)*2)/2}")
                tokens.append(f"{str(element.pitch)}_{duration}")
                prev_offset = element.offset
            elif isinstance(element, chord.Chord):
                if prev_offset >= 0 and element.offset - prev_offset > 0.5:
                    tokens.append(f"REST_{round((element.offset - prev_offset)*2)/2}")
                chord_str = '.'.join(str(n) for n in element.normalOrder)
                tokens.append(f"{chord_str}_{duration}")
    return tokens

# Training
def train_model():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    tokens = get_notes()
    vocab = sorted(set(tokens))
    note_to_int = {n: i for i, n in enumerate(vocab)}
    int_to_note = {i: n for i, n in enumerate(vocab)}

    dataset = MIDIDataset(tokens, note_to_int)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=256, sampler=sampler)

    model = TransformerModel(len(vocab)).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(100):
        sampler.set_epoch(epoch)
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if local_rank == 0:
            print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")
            torch.save(model.module.state_dict(), "transformer_music_ddp.pth")

    if local_rank == 0:
        with open('./data/note_mappings.pkl', 'wb') as f:
            pickle.dump({'note_to_int': note_to_int, 'int_to_note': int_to_note}, f)

# Generate
def generate_music():
    tokens = get_notes()
    vocab = sorted(set(tokens))
    note_to_int = {n: i for i, n in enumerate(vocab)}
    int_to_note = {i: n for i, n in enumerate(vocab)}

    model = TransformerModel(len(vocab)).to(device)
    model.load_state_dict(torch.load("transformer_music_ddp.pth", map_location=device))
    model.eval()

    seed = tokens[np.random.randint(0, len(tokens)-31)][:30]
    pattern = [note_to_int[n] for n in seed]

    output = []
    for _ in range(500):
        inp = torch.tensor(pattern, dtype=torch.long).unsqueeze(0).to(device)
        pred = model(inp).squeeze(0).detach().cpu().numpy()
        pred = np.log(pred + 1e-9) / 1.2
        exp_preds = np.exp(pred)
        pred = exp_preds / np.sum(exp_preds)
        next_index = np.random.choice(len(pred), p=pred)
        output.append(int_to_note[next_index])
        pattern = pattern[1:] + [next_index]

    stream_out = stream.Stream()
    offset = 0
    for token in output:
        try:
            if token.startswith('REST_'):
                offset += float(token.split('_')[1])
                continue
            base, dur = token.rsplit('_', 1)
            duration = float(dur)
            if '.' in base or base.isdigit():
                chord_notes = [note.Note(int(n)) for n in base.split('.')]
                c = chord.Chord(chord_notes)
                c.offset = offset
                c.quarterLength = duration
                stream_out.append(c)
            else:
                n = note.Note(base)
                n.offset = offset
                n.quarterLength = duration
                stream_out.append(n)
            offset += duration
        except:
            continue

    stream_out.write('midi', fp='torch8_transformer_generated.mid')
    print("[INFO] Generated MIDI saved as 'torch8_transformer_generated.mid'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'generate'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    else:
        generate_music()
