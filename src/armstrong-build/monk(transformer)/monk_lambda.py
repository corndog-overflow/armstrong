import argparse
import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream, duration
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding, Add, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from fractions import Fraction

np.random.seed(42)

def get_note_rhythm_tokens():
    print("[INFO] Extracting pitch-duration tokens...")
    tokens = []
    for file in glob.glob("./jazz_and_stuff/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None
        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            dur = element.quarterLength
            if isinstance(element, note.Note):
                tokens.append(f"{element.pitch}_{dur}")
            elif isinstance(element, chord.Chord):
                chord_token = '.'.join(str(n) for n in element.normalOrder)
                tokens.append(f"{chord_token}_{dur}")

    os.makedirs('./data', exist_ok=True)
    with open('./data/tokens', 'wb') as f:
        pickle.dump(tokens, f)

    return tokens

def prepare_sequences(tokens, n_vocab, sequence_length=100):
    print("[INFO] Preparing sequences...")
    pitchnames = sorted(set(item for item in tokens))
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

def create_transformer_model(seq_len, vocab_size, embed_dim=256, num_heads=4, ff_dim=512):
    inputs = Input(shape=(seq_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    ff_output = Dense(ff_dim, activation='relu')(x)
    ff_output = Dense(embed_dim)(ff_output)
    x = Add()([x, ff_output])
    x = LayerNormalization()(x)
    x = Flatten()(x)
    outputs = Dense(vocab_size, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy')
    return model

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds.flatten(), 1)
    return np.argmax(probas)

def train_network():
    tokens = get_note_rhythm_tokens()
    n_vocab = len(set(tokens))
    network_input, network_output, _, _ = prepare_sequences(tokens, n_vocab)

    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"[INFO] Multiple GPUs detected ({num_gpus}). Using MirroredStrategy.")
    else:
        strategy = tf.distribute.get_strategy()
        print(f"[INFO] Single GPU or CPU detected. Using default strategy.")

    base_batch_size = 64
    effective_batch_size = base_batch_size * max(1, num_gpus)

    with strategy.scope():
        model = create_transformer_model(seq_len=network_input.shape[1], vocab_size=n_vocab)

    os.makedirs('./weights', exist_ok=True)
    filepath = "./weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    print(f"[INFO] Training transformer model with effective batch size: {effective_batch_size}")
    model.fit(network_input, network_output, epochs=100, batch_size=effective_batch_size, callbacks=callbacks_list)

def generate_music():
    with open('./data/tokens', 'rb') as f:
        tokens = pickle.load(f)

    n_vocab = len(set(tokens))
    network_input, _, token_to_int, pitchnames = prepare_sequences(tokens, n_vocab)
    int_to_token = dict((number, token) for number, token in enumerate(pitchnames))

    model = create_transformer_model(seq_len=network_input.shape[1], vocab_size=n_vocab)

    weight_files = sorted(glob.glob("./weights/weights-improvement-*.hdf5"))
    if not weight_files:
        print("[ERROR] No weights found!")
        return
    model.load_weights(weight_files[-1])

    os.makedirs('./outputs', exist_ok=True)

    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)
    base_batch_size = 1
    effective_batch_size = base_batch_size * max(1, num_gpus)

    for song_idx in range(5):
        start = np.random.randint(0, len(network_input) - 1)
        pattern = list(network_input[start])
        prediction_output = []

        for _ in range(500):
            prediction_input = np.reshape(pattern, (1, len(pattern)))
            prediction = model.predict(prediction_input, batch_size=effective_batch_size, verbose=0)[0]
            index = sample_with_temperature(prediction, temperature=1.0)
            result = int_to_token[index]
            prediction_output.append(result)
            pattern.append(index)
            pattern = pattern[1:]

        create_midi(prediction_output, song_idx)

def create_midi(prediction_output, idx):
    print(f"[INFO] Creating MIDI file for song {idx}...")
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '_' not in pattern:
            continue
        pitch_part, dur_part = pattern.split('_')
        dur = float(Fraction(dur_part))
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

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=f'./outputs/transformer_generated_{idx}.mid')
    print(f"[INFO] Saved to ./outputs/transformer_generated_{idx}.mid")

def main():
    parser = argparse.ArgumentParser(description="Transformer-based Music Generator with Rhythm Tokens")
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True)
    args = parser.parse_args()

    if args.mode == 'train':
        train_network()
    elif args.mode == 'generate':
        generate_music()
    else:
        raise ValueError("Invalid mode")

if __name__ == '__main__':
    main()



