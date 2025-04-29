import glob
import pickle
import numpy as np
import os
from music21 import converter, instrument, note, chord, stream
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Set mixed precision for A100 GPUs
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Color codes
RESET = '\033[0m'
ORANGE = '\033[38;5;208m'
RUST = '\033[38;5;166m'
MAROON = '\033[38;5;88m'


def get_notes(directory="./jazz_and_stuff", train=True):
    print(f"{ORANGE}[INFO]{RESET} Extracting notes and chords...")
    tokens = []
    if not train:
        try:
            with open('./data/tokens', 'rb') as f:
                tokens = pickle.load(f)
                return tokens
        except FileNotFoundError:
            print(f"{RUST}[WARN]{RESET} Token file not found. Extracting from MIDI.")

    for file in glob.glob(directory + "/*.mid"):
        midi = converter.parse(file)
        try:
            notes = instrument.partitionByInstrument(midi).parts[0].recurse()
        except:
            notes = midi.flat.notes

        note_offsets = []
        for el in notes:
            if isinstance(el, (note.Note, chord.Chord)):
                note_offsets.append(el.offset)

        note_offsets.sort()
        prev_offset = -1
        for el in notes:
            duration = round(el.quarterLength * 2) / 2
            if isinstance(el, note.Note):
                if prev_offset >= 0 and el.offset - prev_offset > 0.5:
                    tokens.append(f"REST_{round((el.offset - prev_offset)*2)/2}")
                tokens.append(f"{str(el.pitch)}_{duration}")
                prev_offset = el.offset
            elif isinstance(el, chord.Chord):
                if prev_offset >= 0 and el.offset - prev_offset > 0.5:
                    tokens.append(f"REST_{round((el.offset - prev_offset)*2)/2}")
                chord_str = '.'.join(str(n) for n in el.normalOrder)
                tokens.append(f"{chord_str}_{duration}")
                prev_offset = el.offset

    with open('./data/tokens', 'wb') as f:
        pickle.dump(tokens, f)
    return tokens


def prepare_sequences(notes, n_vocab, seq_len=30, train=True):
    print(f"{ORANGE}[INFO]{RESET} Preparing sequences...")
    pitches = sorted(set(notes))
    note_to_int = dict((n, i) for i, n in enumerate(pitches))
    int_to_note = dict((i, n) for i, n in enumerate(pitches))

    if train:
        net_in, net_out = [], []
        for i in range(len(notes) - seq_len):
            net_in.append([note_to_int[n] for n in notes[i:i + seq_len]])
            net_out.append(note_to_int[notes[i + seq_len]])
        return np.array(net_in), tf.keras.utils.to_categorical(net_out, num_classes=n_vocab), note_to_int, int_to_note
    else:
        net_in = []
        for i in range(len(notes) - seq_len):
            net_in.append([note_to_int[n] for n in notes[i:i + seq_len]])
        return np.array(net_in), note_to_int, int_to_note


def positional_encoding(position, d_model):
    angles = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    pos = np.arange(position)[:, np.newaxis]
    angle_rads = pos * angles
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], tf.float32)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(inputs + x)
    ffn = Dense(ff_dim, activation='relu')(x)
    ffn = Dropout(dropout)(ffn)
    ffn = Dense(inputs.shape[-1])(ffn)
    return LayerNormalization(epsilon=1e-6)(x + ffn)


def build_transformer(vocab_size, seq_len=30, embed_dim=256):
    inputs = Input(shape=(seq_len,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(vocab_size, embed_dim)(inputs)
    x += positional_encoding(seq_len, embed_dim)
    for _ in range(3):
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=512)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(vocab_size, activation='softmax', dtype='float32')(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001))
    return model


def train_network():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        notes = get_notes()
        vocab_size = len(set(notes))
        net_in, net_out, note2int, int2note = prepare_sequences(notes, vocab_size)
        model = build_transformer(vocab_size)

    model.summary()
    callbacks = [
        ModelCheckpoint('weights_transformer_checkpoint.keras', monitor='loss', save_best_only=True),
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)
    ]
    model.fit(net_in, net_out, batch_size=512, epochs=300, callbacks=callbacks)
    with open('./data/note_mappings.pkl', 'wb') as f:
        pickle.dump({'note_to_int': note2int, 'int_to_note': int2note}, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'generate'], default='train')
    args = parser.parse_args()
    if args.mode == 'train':
        train_network()
    else:
        print("[TODO] Generation mode not yet supported in this multi-GPU version.")
