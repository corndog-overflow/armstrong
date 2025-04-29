import argparse
import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import to_categorical  # <<== 修正点
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

# 保证 reproducibility
np.random.seed(42)

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

    os.makedirs('./data', exist_ok=True)
    with open('./data/tokens', 'wb') as f:
        pickle.dump(notes, f)

    return notes

def prepare_sequences(notes, n_vocab):
    print("[INFO] Preparing sequences...")
    sequence_length = 100
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)

    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output)  # <<== 修正点

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    print("[INFO] Creating model...")
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def train_network():
    notes = get_notes()
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)

    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"[INFO] Multiple GPUs detected ({num_gpus}). Using MirroredStrategy for distributed training.")
    else:
        strategy = tf.distribute.get_strategy()  # 默认单卡策略
        print(f"[INFO] Single GPU or CPU detected. Using default strategy.")

    base_batch_size = 64
    effective_batch_size = base_batch_size * num_gpus if num_gpus > 0 else base_batch_size

    with strategy.scope():
        model = create_network(network_input, n_vocab)

    os.makedirs('./weights', exist_ok=True)
    filepath = "./weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    print(f"[INFO] Training model with effective batch size: {effective_batch_size}")
    model.fit(network_input, network_output, epochs=200, batch_size=effective_batch_size, callbacks=callbacks_list)


def generate_music():
    print("[INFO] Loading notes...")
    with open('./data/tokens', 'rb') as f:
        notes = pickle.load(f)

    n_vocab = len(set(notes))
    pitchnames = sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    network_input, _ = prepare_sequences(notes, n_vocab)
    model = create_network(network_input, n_vocab)

    # Load the best weights
    weight_files = sorted(glob.glob("./weights/weights-improvement-*.hdf5"))
    if not weight_files:
        print("[ERROR] No weights found! Please train the model first.")
        return
    latest_weight = weight_files[-1]
    print(f"[INFO] Loading weights from {latest_weight}")
    model.load_weights(latest_weight)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print("[INFO] Generating music...")
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    prediction_output = []

    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, pattern.shape[0], 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        # pattern逐步更新，同时归一化
        pattern = np.append(pattern, index / float(n_vocab))
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
    midi_stream.write('midi', fp='./outputs/transformer_generated.mid')
    print("[INFO] MIDI file created at ./outputs/transformer_generated.mid")

def main():
    parser = argparse.ArgumentParser(description="Monk Lambda Music LSTM Generator")
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True,
                        help="Choose 'train' to train the model or 'generate' to generate music.")
    args = parser.parse_args()

    if args.mode == 'train':
        train_network()
    elif args.mode == 'generate':
        generate_music()
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'generate'.")

if __name__ == '__main__':
    main()


