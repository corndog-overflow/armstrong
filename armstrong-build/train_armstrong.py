import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import os

# ANSI escape color codes - Fall colors palette
ORANGE = '\033[38;5;208m'  # Bright orange
RUST = '\033[38;5;166m'    # Rust/deep orange
BROWN = '\033[38;5;130m'   # Brown
GOLD = '\033[38;5;220m'    # Gold/amber
MAROON = '\033[38;5;88m'   # Deep maroon/burgundy
RESET = '\033[0m'          # Reset to default
BOLD = '\033[1m'           # Bold text

def get_notes(directory="./jazz_and_stuff", train=True):
    print(f"{ORANGE}[INFO]{RESET} Extracting notes and chords with durations from MIDI files...")
    tokens = []

    for file in glob.glob(directory + "/*.mid"):
        print(f"{RUST}[INFO]{RESET} Parsing file: {file}")
        midi = converter.parse(file)

        try:
            parse_instrums = instrument.partitionByInstrument(midi)
            notes_to_parse = parse_instrums.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        note_offsets = []
        for element in notes_to_parse:
            if isinstance(element, (note.Note, chord.Chord)):
                note_offsets.append(element.offset)

        note_offsets.sort()
        prev_offset = -1
        for element in notes_to_parse:
            duration = round(element.quarterLength * 2) / 2  # round to nearest 0.5

            if isinstance(element, note.Note):
                if prev_offset >= 0 and element.offset - prev_offset > 0.5:
                    rest_duration = round((element.offset - prev_offset) * 2) / 2
                    tokens.append(f"REST_{rest_duration}")
                pitch = str(element.pitch)
                tokens.append(f"{pitch}_{duration}")
                prev_offset = element.offset

            elif isinstance(element, chord.Chord):
                if prev_offset >= 0 and element.offset - prev_offset > 0.5:
                    rest_duration = round((element.offset - prev_offset) * 2) / 2
                    tokens.append(f"REST_{rest_duration}")
                chord_str = '.'.join(str(n) for n in element.normalOrder)
                tokens.append(f"{chord_str}_{duration}")
                prev_offset = element.offset

    print(f"{ORANGE}[INFO]{RESET} Total tokens extracted: {len(tokens)}")
    file_path = './data/tokens'
    dir_path = os.path.dirname(file_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(file_path, 'wb') as filepath:
        pickle.dump(tokens, filepath)

    print(f"{ORANGE}[INFO]{RESET} Tokens saved to './data/tokens'")
    return tokens


def prepare_sequences_train(notes, n_vocab):
    print(f"{ORANGE}[INFO]{RESET} Preparing input/output sequences...")
    sequence_length = 30

    pitches = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitches))

    net_in = []
    net_out = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        net_in.append([note_to_int[char] for char in sequence_in])
        net_out.append(note_to_int[sequence_out])

    n_patterns = len(net_in)

    net_in = numpy.reshape(net_in, (n_patterns, sequence_length, 1))
    net_in = net_in / float(n_vocab)
    net_out = tf.keras.utils.to_categorical(net_out, num_classes=n_vocab)

    print(f"{ORANGE}[INFO]{RESET} Total patterns: {n_patterns}")
    return (net_in, net_out)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization as BatchNorm, Activation, Add
from tensorflow.keras.optimizers import Adam

def layer_model(network_input, n_vocab):
    print(f"{ORANGE}[INFO]{RESET} creating armstrong model")

    input_shape = (network_input.shape[1], network_input.shape[2])
    inputs = Input(shape=input_shape)

    x1 = LSTM(512, return_sequences=True, recurrent_dropout=0.3)(inputs)
    x2 = LSTM(512, return_sequences=True, recurrent_dropout=0.2)(x1)
    x = Add()([x1, x2]) 
    x3 = LSTM(384, recurrent_dropout=0.2)(x)
    x3 = BatchNorm()(x3)
    x3 = Dropout(0.4)(x3)
    dense1 = Dense(256)(x3)
    dense1 = Activation('relu')(dense1)
    dense1 = BatchNorm()(dense1)
    dense1 = Dropout(0.4)(dense1)
    x_proj = Dense(256)(x3) #added another residual connection
    x = Add()([x_proj, dense1])
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(n_vocab)(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy', optimizer=Adam())

    return model


def train(model, network_input, network_output, finetune=False):
    print(f"{ORANGE}[INFO]{RESET} Starting training...")
    filepath = "weights_checkpoint.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    if finetune:
        print(f"{GOLD}[INFO]{RESET} Loading weights for fine-tuning...")
        model.load_weights('weights_checkpoint.keras')
    
    history = model.fit(network_input, network_output, epochs=550, batch_size=128, callbacks=callbacks_list)
    final_loss = history.history['loss'][-1]
    with open('./loss.txt', 'w') as file:
        file.write(str(final_loss))
    
    print(f"{ORANGE}[INFO]{RESET} Final loss value ({final_loss}) saved to 'loss.txt'.")
    print(f"{RUST}[INFO]{RESET} Training complete.")

def train_network():
    print(f"{MAROON}+=======================================+{RESET}")
    print(f"{MAROON}|  Starting Armstrong Training Pipeline |{RESET}")
    print(f"{MAROON}+=======================================+{RESET}")
    notes = get_notes()
    n_vocab = len(set(notes))
    print(f"{ORANGE}[INFO]{RESET} Vocabulary size: {n_vocab}")
    
    network_input, network_output = prepare_sequences_train(notes, n_vocab)
    
    model = layer_model(network_input, n_vocab)
    
    train(model, network_input, network_output, True)

print(f"{ORANGE}Attempting to train Armstrong...{RESET} \n")
train_network()
print(f"{RUST}Armstrong training complete.{RESET}")
