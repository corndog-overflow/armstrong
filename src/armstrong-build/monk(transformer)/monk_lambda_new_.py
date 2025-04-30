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
                    chord_token = '.'.join(str(n) for n in element.normalOrder)
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
def jazz_chord_reward(token):
    if '_' not in token:
        return 0.0
    pitch_part = token.split('_')[0]
    if '.' in pitch_part:
        notes = [int(n) % 12 for n in pitch_part.split('.')]
        if len(notes) >= 3:
            if (7 in notes) or (10 in notes) or (2 in notes):
                return 1.0
    return 0.0

def pitch_distance_reward(seq):
    reward = 0.0
    prev_pitch = None
    for token in seq:
        if '_' not in token:
            continue
        pitch_part = token.split('_')[0]
        if '.' not in pitch_part:
            try:
                curr_pitch = pitch.Pitch(pitch_part).midi
                if prev_pitch is not None:
                    dist = abs(curr_pitch - prev_pitch)
                    if dist <= 2:
                        reward += 1.0
                    elif dist > 7:
                        reward -= 0.5
                prev_pitch = curr_pitch
            except:
                continue
    return reward / max(1, len(seq))

def rhythm_reward(seq):
    reward = 0.0
    durations = []
    for token in seq:
        if '_' not in token:
            continue
        dur = float(Fraction(token.split('_')[1]))
        durations.append(dur)
    unique_durs = set(durations)
    if 0.5 in unique_durs or 1.5 in unique_durs:
        reward += 1.0
    if 0.75 in unique_durs or 1.75 in unique_durs:
        reward += 0.5
    return reward

def compute_reward(sequence, int_to_token):
    tokens = [int_to_token[i] for i in sequence]
    chord_r = sum(jazz_chord_reward(t) for t in tokens) / max(1, len(tokens))
    pitch_r = pitch_distance_reward(tokens)
    rhythm_r = rhythm_reward(tokens)
    return (chord_r * 0.4) + (pitch_r * 0.4) + (rhythm_r * 0.2)

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

def create_transformer_model(seq_len, vocab_size, embed_dim=256, num_heads=4, ff_dim=512, num_layers=3):
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
def generate_sequence(model, seed_input, length=50, temperature=0.8):
    sequence = seed_input.copy()
    for _ in range(length):
        preds = model.predict(np.array([sequence[-100:]]), verbose=0)[0]
        next_token = np.random.choice(len(preds), p=preds)
        sequence.append(next_token)
    return sequence

def reinforce_update(model, sequences, rewards, optimizer):
    with tf.GradientTape() as tape:
        loss = 0.0
        for seq, reward in zip(sequences, rewards):
            input_seq = seq[:-1][-100:]  
            target_token = seq[-1]      

            # logits: [1, vocab_size]
            logits = model(np.array([input_seq]), training=True)[0]

            # 计算对 target_token 的 loss
            neg_logprob = tf.keras.losses.sparse_categorical_crossentropy(
                [target_token], [logits], from_logits=False
            )
            loss += neg_logprob * (1.0 - reward)

        loss /= len(sequences)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def train_with_rl(model, network_input, network_output, token_to_int, epochs=30, rl_interval=3):
    optimizer = Adam(learning_rate=0.0001)
    int_to_token = {v: k for k, v in token_to_int.items()}
    gpus = tf.config.list_physical_devices('GPU')
    gpu_count = len(gpus) if gpus else 1
    seq_per_epoch = 16 * gpu_count
    print(f"[INFO] RL Training with {seq_per_epoch} sequences per update (using {gpu_count} GPU(s))")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.fit(network_input, network_output, batch_size=64 * gpu_count, verbose=1)

        if epoch % rl_interval == 0:
            print("RL phase - Generating sequences...")
            sequences = []
            rewards = []
            for _ in range(seq_per_epoch):
                seed_idx = np.random.randint(0, len(network_input) - 1)
                seed = list(network_input[seed_idx])
                seq = generate_sequence(model, seed, length=50)
                sequences.append(seq)
                rewards.append(compute_reward(seq, int_to_token))

            avg_reward = np.mean(rewards)
            print(f"Average reward: {avg_reward:.3f}")
            reinforce_update(model, sequences, rewards, optimizer)
            if avg_reward > 0.6:
                model.save_weights(f"./weights/rl_best_{avg_reward:.3f}.h5")
def train_network():
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./weights', exist_ok=True)
    tokens = get_note_rhythm_tokens()
    n_vocab = len(set(tokens))
    network_input, network_output, token_to_int, _ = prepare_sequences(tokens, n_vocab)
    gpus = tf.config.list_physical_devices('GPU')
    gpu_count = len(gpus) if gpus else 1
    batch_size = 64 * gpu_count
    print(f"[INFO] Detected {gpu_count} GPU(s), using batch size = {batch_size}")
    model = create_transformer_model(seq_len=network_input.shape[1], vocab_size=n_vocab, embed_dim=256, num_heads=8, ff_dim=512, num_layers=3)
    print("\nStarting supervised training...")
    model.fit(network_input, network_output, epochs=3, batch_size=batch_size)
    print("\nStarting RL fine-tuning...")
    train_with_rl(model, network_input, network_output, token_to_int, epochs=30, rl_interval=3)

def generate_music():
    with open('./data/tokens', 'rb') as f:
        tokens = pickle.load(f)
    n_vocab = len(set(tokens))
    network_input, _, token_to_int, pitchnames = prepare_sequences(tokens, n_vocab)
    int_to_token = {v: k for k, v in token_to_int.items()}
    model = create_transformer_model(seq_len=network_input.shape[1], vocab_size=n_vocab)
    weight_files = glob.glob("./weights/rl_best_*.h5")
    if weight_files:
        model.load_weights(sorted(weight_files)[-1])
        print(f"Loaded weights: {weight_files[-1]}")
    else:
        model.load_weights(sorted(glob.glob("./weights/*.h5"))[-1])
    os.makedirs('./outputs', exist_ok=True)
    for i in range(3):
        seed_idx = np.random.randint(0, len(network_input) - 1)
        generated = generate_sequence(model, list(network_input[seed_idx]), length=200)
        create_midi(generated, i, int_to_token)

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
            notes = [note.Note(int(p)) for p in pitch_part.split('.')]
            new_chord = chord.Chord(notes)
            new_chord.duration = duration.Duration(dur)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pitch_part)
            new_note.duration = duration.Duration(dur)
            new_note.offset = offset
            output_notes.append(new_note)
        offset += dur
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=f'./outputs/jazz_generated_{idx}.mid')

def main():
    parser = argparse.ArgumentParser(description="Jazz Music Transformer with RL")
    parser.add_argument('--mode', choices=['train', 'generate'], required=True)
    args = parser.parse_args()
    if args.mode == 'train':
        train_network()
    elif args.mode == 'generate':
        generate_music()

if __name__ == '__main__':
    main()
