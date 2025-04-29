import argparse
import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from music21 import converter, instrument, note, chord, stream

# 保证 reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 设置device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1])
        return x

class MIDIDataset(Dataset):
    def __init__(self, notes, note_to_int, sequence_length=100):
        self.sequence_length = sequence_length
        self.notes = notes
        self.note_to_int = note_to_int

    def __len__(self):
        return len(self.notes) - self.sequence_length

    def __getitem__(self, idx):
        seq_in = self.notes[idx:idx + self.sequence_length]
        seq_out = self.notes[idx + self.sequence_length]
        x = torch.tensor([self.note_to_int[n] for n in seq_in], dtype=torch.long)
        y = torch.tensor(self.note_to_int[seq_out], dtype=torch.long)
        return x, y

def get_notes():
    print("[INFO] Extracting notes and chords...")
    notes = []
    for file in glob.glob("./jazz_and_stuff/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None
        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

def train_model():
    tokens = get_notes()
    vocab = sorted(set(tokens))
    note_to_int = {note: i for i, note in enumerate(vocab)}
    int_to_note = {i: note for i, note in enumerate(vocab)}

    dataset = MIDIDataset(tokens, note_to_int)

    # 检测GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"[INFO] GPUs detected: {num_gpus}")

    base_batch_size = 128
    effective_batch_size = base_batch_size * num_gpus if num_gpus > 0 else base_batch_size

    loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = TransformerModel(len(vocab)).to(device)

    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"[INFO] Using DataParallel with {num_gpus} GPUs.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(100):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

        save_model = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(save_model.state_dict(), "./outputs/transformer_music.pth")

    os.makedirs('./data', exist_ok=True)
    with open('./data/note_mappings.pkl', 'wb') as f:
        pickle.dump({'note_to_int': note_to_int, 'int_to_note': int_to_note}, f)

def generate_music():
    print("[INFO] Generating music...")
    with open('./data/note_mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)

    note_to_int = mappings['note_to_int']
    int_to_note = mappings['int_to_note']

    vocab_size = len(note_to_int)
    model = TransformerModel(vocab_size).to(device)

    # 加载权重
    checkpoint = torch.load("./outputs/transformer_music.pth", map_location=device)
    model.load_state_dict(checkpoint)

    model.eval()

    start = np.random.randint(0, len(note_to_int) - 100)
    pattern = list(note_to_int.keys())[start:start + 100]
    pattern = [note_to_int[n] for n in pattern]

    prediction_output = []

    for note_index in range(500):
        input_seq = torch.tensor(pattern, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(input_seq)
        idx = torch.argmax(prediction, dim=1).item()
        result = int_to_note[idx]

        prediction_output.append(result)
        pattern.append(idx)
        pattern = pattern[1:]

    create_midi(prediction_output)

def create_midi(prediction_output):
    print("[INFO] Creating MIDI file...")
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    os.makedirs('./outputs', exist_ok=True)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='./outputs/transformer_generated_torch.mid')


def main():
    parser = argparse.ArgumentParser(description="Monk Torch Music Transformer")
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True,
                        help="Choose 'train' to train the model or 'generate' to generate music.")
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'generate':
        generate_music()
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'generate'.")

if __name__ == '__main__':
    main()

