import argparse
import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream, duration, pitch
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding, Add, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
import tensorflow as tf
from fractions import Fraction
import random

np.random.seed(42)

def get_note_rhythm_tokens():
    print("[INFO] Extracting pitch-duration tokens...")
    tokens = []
    for file in glob.glob("./jazz_and_stuff/*.mid"):
        try:
            midi = converter.parse(file)
            try:
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                dur = element.quarterLength
                if dur <= 0 or dur > 8:
                    continue

                if isinstance(element, note.Note):
                    tokens.append(f"{element.pitch}_{dur}")
                elif isinstance(element, chord.Chord):
                    chord_token = '.'.join(str(p.midi) for p in element.pitches)
                    tokens.append(f"{chord_token}_{dur}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    with open('./data/tokens', 'wb') as f:
        pickle.dump(tokens, f)

    return tokens

def prepare_sequences(tokens, n_vocab, sequence_length=100):
    pitchnames = sorted(set(tokens))
    token_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []
    for i in range(0, len(tokens) - sequence_length, 1):
        seq_in = tokens[i:i + sequence_length]
        seq_out = tokens[i + sequence_length]
        network_input.append([token_to_int[char] for char in seq_in])
        network_output.append(token_to_int[seq_out])
    network_input = np.array(network_input)
    network_output = to_categorical(network_output, num_classes=n_vocab)
    return network_input, network_output, token_to_int, pitchnames

def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_block(x, embed_dim, num_heads, ff_dim, rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(x, x)
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def create_transformer_model(seq_len, vocab_size, embed_dim=256, num_heads=8, ff_dim=512, num_layers=3):
    inputs = Input(shape=(seq_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    pos_encoding = positional_encoding(seq_len, embed_dim)
    x = x + pos_encoding
    for _ in range(num_layers):
        x = transformer_block(x, embed_dim, num_heads, ff_dim)
    x = Flatten()(x)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(vocab_size, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy')
    return model

def generate_sequence_batch(model, seeds, length=50):
    sequences = [seed.copy() for seed in seeds]
    for _ in range(length):
        input_batch = np.array([seq[-100:] for seq in sequences])
        preds_batch = model.predict(input_batch, verbose=0)
        for i, preds in enumerate(preds_batch):
            next_token = np.random.choice(len(preds), p=preds)
            sequences[i].append(next_token)
    return sequences

def create_midi(sequence, idx, int_to_token):
    offset = 0
    output_notes = []
    for token in sequence:
        token_str = int_to_token.get(token, "")
        if '_' not in token_str:
            continue

        pitch_part, dur_part = token_str.split('_')
        dur = float(Fraction(dur_part))

        if '.' in pitch_part:
            try:
                notes = [note.Note(int(p)) for p in pitch_part.split('.')]
                new_chord = chord.Chord(notes)
                new_chord.duration = duration.Duration(dur)
                new_chord.offset = offset
                output_notes.append(new_chord)
            except Exception as e:
                print(f"[WARNING] Skipped invalid chord '{pitch_part}': {e}")
        else:
            try:
                if pitch_part.isdigit():
                    raise ValueError(f"'{pitch_part}' is not a valid note name")
                new_note = note.Note(pitch_part)
                new_note.duration = duration.Duration(dur)
                new_note.offset = offset
                output_notes.append(new_note)
            except Exception as e:
                print(f"[WARNING] Skipped invalid note '{pitch_part}': {e}")

        offset += dur

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=f'./outputs/jazz_generated_{idx}.mid')

def generate_music():
    with open('./data/tokens', 'rb') as f:
        tokens = pickle.load(f)
    n_vocab = len(set(tokens))
    network_input, _, token_to_int, pitchnames = prepare_sequences(tokens, n_vocab)
    int_to_token = {v: k for k, v in token_to_int.items()}

    model = create_transformer_model(seq_len=network_input.shape[1],
                                     vocab_size=n_vocab,
                                     embed_dim=256,
                                     num_heads=8,
                                     ff_dim=512,
                                     num_layers=3)
    model.summary()

    weight_path = "./weights/rl_best.h5"
    if os.path.exists(weight_path):
        model.load_weights(weight_path)
        print(f"[INFO] Loaded best RL weights: {weight_path}")
    else:
        fallback_weights = sorted(glob.glob("./weights/*.h5"))
        if fallback_weights:
            model.load_weights(fallback_weights[-1])
            print(f"[INFO] Loaded fallback weights: {fallback_weights[-1]}")
        else:
            raise FileNotFoundError("[ERROR] No weight files found in ./weights/. Please run training first.")

    os.makedirs('./outputs', exist_ok=True)
    for i in range(3):
        seed_idx = np.random.randint(0, len(network_input) - 1)
        seed = list(network_input[seed_idx])
        generated = generate_sequence_batch(model, [seed], length=200)[0]
        create_midi(generated, i, int_to_token)

def main():
    parser = argparse.ArgumentParser(description="Jazz Music Transformer with RL")
    parser.add_argument('--mode', choices=['train', 'generate'], required=True)
    args = parser.parse_args()

    if args.mode == 'train':
        tokens = get_note_rhythm_tokens()
        n_vocab = len(set(tokens))
        network_input, network_output, token_to_int, _ = prepare_sequences(tokens, n_vocab)

        gpus = tf.config.list_physical_devices('GPU')
        gpu_count = len(gpus) if gpus else 1
        batch_size = 64 * gpu_count
        print(f"[INFO] Detected {gpu_count} GPU(s), using batch size = {batch_size}")

        model = create_transformer_model(seq_len=network_input.shape[1], vocab_size=n_vocab,
                                         embed_dim=256, num_heads=8, ff_dim=512, num_layers=3)

        print("\n[INFO] Starting supervised training...")
        model.fit(network_input, network_output, epochs=300, batch_size=batch_size)

        os.makedirs('./weights', exist_ok=True)
        model.save_weights("./weights/final_supervised.h5")
        print("[INFO] Saved final supervised model to ./weights/final_supervised.h5")

    elif args.mode == 'generate':
        generate_music()

if __name__ == '__main__':
    main()

