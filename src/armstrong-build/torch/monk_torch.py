import argparse
import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from music21 import converter, instrument, note, chord, stream, duration
from fractions import Fraction
from collections import Counter

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(0.1)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x)

class MIDIDataset(Dataset):
    def __init__(self, tokens, token_to_int, sequence_length=100):
        self.tokens = tokens
        self.token_to_int = token_to_int
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.tokens) - self.sequence_length

    def __getitem__(self, idx):
        seq_in = self.tokens[idx:idx + self.sequence_length]
        seq_out = self.tokens[idx + 1:idx + self.sequence_length + 1]
        x = torch.tensor([self.token_to_int[n] for n in seq_in], dtype=torch.long)
        y = torch.tensor([self.token_to_int[n] for n in seq_out], dtype=torch.long)
        return x, y

def discretize_duration(dur):
    return round(float(dur) * 8) / 8

def get_note_rhythm_tokens():
    print("[INFO] Extracting pitch-duration tokens...")
    tokens = []
    for file in glob.glob("./jazz_and_stuff/*.mid"):
        midi = converter.parse(file)
        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            dur = element.quarterLength
            try:
                dur = discretize_duration(dur)
                dur_str = str(Fraction(dur).limit_denominator(8))
            except:
                continue

            if isinstance(element, note.Note):
                tokens.append(f"{element.pitch}_{dur_str}")
            elif isinstance(element, chord.Chord):
                chord_token = '.'.join(str(n) for n in element.normalOrder)
                tokens.append(f"{chord_token}_{dur_str}")

    counter = Counter(tokens)
    filtered_tokens = [t for t in tokens if counter[t] >= 5]

    print(f"[INFO] Original vocab size: {len(set(tokens))}")
    print(f"[INFO] Filtered vocab size: {len(set(filtered_tokens))}")

    os.makedirs('./data', exist_ok=True)
    with open('./data/tokens', 'wb') as f:
        pickle.dump(filtered_tokens, f)

    return filtered_tokens

def train_network():
    tokens = get_note_rhythm_tokens()
    vocab = sorted(set(tokens))
    token_to_int = {note: i for i, note in enumerate(vocab)}
    int_to_token = {i: note for i, note in enumerate(vocab)}

    dataset = MIDIDataset(tokens, token_to_int)
    num_gpus = torch.cuda.device_count()
    print(f"[INFO] GPUs detected: {num_gpus}")

    base_batch_size = 64
    effective_batch_size = base_batch_size * max(1, num_gpus)

    loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = TransformerModel(len(vocab)).to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"[INFO] Using DataParallel with {num_gpus} GPUs.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs('./weights', exist_ok=True)

    model.train()
    for epoch in range(100):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

        save_model = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({
            'model_state_dict': save_model.state_dict(),
            'token_to_int': token_to_int,
            'int_to_token': int_to_token
        }, f"./weights/epoch_{epoch+1}.pth")

def sample_with_temperature(logits, temperature=1.0):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return idx.item()

def generate_music(temperature=1.0):
    print(f"[INFO] Generating music with temperature={temperature}")
    checkpoint = torch.load(sorted(glob.glob("./weights/epoch_*.pth"))[-1], map_location=device)
    token_to_int = checkpoint['token_to_int']
    int_to_token = checkpoint['int_to_token']

    vocab_size = len(token_to_int)
    model = TransformerModel(vocab_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    start_idx = np.random.randint(0, len(token_to_int) - 100)
    pattern = list(token_to_int.values())[start_idx:start_idx + 100]
    prediction_output = []
    repeat_count = 0
    last_token = None

    for note_index in range(500):
        input_seq = torch.tensor(pattern, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(input_seq)
        logits = prediction[0, -1]
        idx = sample_with_temperature(logits, temperature)

        if idx == last_token:
            repeat_count += 1
            if repeat_count >= 3:
                logits[idx] = -float('inf')
                idx = sample_with_temperature(logits, temperature)
                repeat_count = 0
        else:
            repeat_count = 0
            last_token = idx

        result = int_to_token[idx]
        prediction_output.append(result)
        pattern.append(idx)
        pattern = pattern[1:]

    create_midi(prediction_output)

def create_midi(prediction_output):
    print("[INFO] Creating MIDI file...")
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '_' not in pattern:
            continue
        pitch_part, dur_part = pattern.split('_')
        try:
            dur = float(Fraction(dur_part))
        except:
            print(f"[WARNING] Invalid duration: {dur_part}")
            continue
        try:
            if '.' in pitch_part or pitch_part.isdigit():
                notes_in_chord = pitch_part.split('.')
                notes_list = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.duration = duration.Duration(dur)
                    new_note.storedInstrument = instrument.Piano()
                    notes_list.append(new_note)
                new_chord = chord.Chord(notes_list)
                new_chord.offset = offset
                output_notes.append(new_chord)
            else:
                new_note = note.Note(pitch_part)
                new_note.duration = duration.Duration(dur)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
        except:
            continue
        offset += dur

    os.makedirs('./outputs', exist_ok=True)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='./outputs/transformer_generated_torch.mid')
    print(f"[INFO] Saved to ./outputs/transformer_generated_torch.mid")

def main():
    parser = argparse.ArgumentParser(description="Torch Transformer Music Generator")
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True)
    parser.add_argument('--temp', type=float, default=1.0, help="Sampling temperature (default=1.0)")
    args = parser.parse_args()

    if args.mode == 'train':
        train_network()
    elif args.mode == 'generate':
        generate_music(temperature=args.temp)
    else:
        raise ValueError("Invalid mode")

if __name__ == '__main__':
    main()
