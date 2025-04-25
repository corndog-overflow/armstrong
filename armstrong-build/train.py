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

# ANSI escape color codes
CYAN = '\e[0;31m'
GREEN = '\e[0;31m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
RESET = '\e[0;31m'
BOLD = '\e[0;31m'

def get_notes(directory="./jazz_and_stuff", train=True):
    print(f"{CYAN}[INFO]{RESET} Extracting notes and chords from MIDI files...")
    tokens = []

    for file in glob.glob(directory + "/*.mid"):
        print(f"{GREEN}[INFO]{RESET} Parsing file: {file}")
        midi = converter.parse(file)

        try:
            parse_instrums = instrument.partitionByInstrument(midi)
            notes_to_parse = parse_instrums.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        note_offsets = []
        for element in notes_to_parse:
            if isinstance(element, note.Note) or isinstance(element, chord.Chord):
                note_offsets.append(element.offset)

        note_offsets.sort()
        prev_offset = -1
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                if prev_offset >= 0 and element.offset - prev_offset > 0.5:
                    rest_duration = round((element.offset - prev_offset) * 2) / 2
                    tokens.append(f"REST_{rest_duration}")
                tokens.append(str(element.pitch))
                prev_offset = element.offset

            elif isinstance(element, chord.Chord):
                if prev_offset >= 0 and element.offset - prev_offset > 0.5:
                    rest_duration = round((element.offset - prev_offset) * 2) / 2
                    tokens.append(f"REST_{rest_duration}")
                tokens.append('.'.join(str(n) for n in element.normalOrder))
                prev_offset = element.offset

    print(f"{CYAN}[INFO]{RESET} Total tokens extracted: {len(tokens)}")
    file_path = './data/tokens'
    dir_path = os.path.dirname(file_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open('./data/tokens', 'wb') as filepath:
        pickle.dump(tokens, filepath)
    print(f"{CYAN}[INFO]{RESET} Tokens saved to './data/tokens'")
    return tokens

def prepare_sequences_train(notes, n_vocab):
    print(f"{CYAN}[INFO]{RESET} Preparing input/output sequences...")
    sequence_length = 20

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

    print(f"{CYAN}[INFO]{RESET} Total patterns: {n_patterns}")
    return (net_in, net_out)

def layer_model(network_input, n_vocab):
    print(f"{CYAN}[INFO]{RESET} Creating LSTM model...")
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]),
                   recurrent_dropout=0.5, return_sequences=True))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    x = LSTM(512, return_sequences=True, recurrent_dropout=0.3)
    model.add(LSTM(512))(x)#adding residual connection
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('relu'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam())

    print(f"{CYAN}[INFO]{RESET} Model compiled successfully.")
    return model

def train(model, network_input, network_output, finetune=False):
    print(f"{CYAN}[INFO]{RESET} Starting training...")
    filepath = "weights_checkpoint.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    if finetune:
        print(f"{YELLOW}[INFO]{RESET} Loading weights for fine-tuning...")
        model.load_weights('weights_checkpoint.keras')

    model.fit(network_input, network_output, epochs=250, batch_size=128, callbacks=callbacks_list)
    print(f"{GREEN}[INFO]{RESET} Training complete.")

def train_network():
    print(f"{MAGENTA}+=======================================+{RESET}")
    print(f"{MAGENTA}|  Starting Armstrong Training Pipeline |{RESET}")
    print(f"{MAGENTA}+=======================================+{RESET}")
    notes = get_notes()
    n_vocab = len(set(notes))
    print(f"{CYAN}[INFO]{RESET} Vocabulary size: {n_vocab}")
    
    network_input, network_output = prepare_sequences_train(notes, n_vocab)
    
    model = layer_model(network_input, n_vocab)
    
    train(model, network_input, network_output, False)

print(f"{CYAN}Attempting to train Armstrong...{RESET} \n")
train_network()
print(f"{GREEN}Armstrong training complete.{RESET}")
