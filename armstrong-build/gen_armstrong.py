import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord, stream
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

def get_notes(directory="./jazz_and_stuff", train=False):
    print(f"{ORANGE}[INFO]{RESET} Extracting notes and chords from MIDI files...")
    tokens = []

    if not train:
        # If not training, just load the tokens from file
        print(f"{ORANGE}[INFO]{RESET} Loading tokens from file for inference...")
        try:
            with open('./data/tokens', 'rb') as filepath:
                tokens = pickle.load(filepath)
            print(f"{ORANGE}[INFO]{RESET} Loaded {len(tokens)} tokens from file.")
            return tokens
        except FileNotFoundError:
            print(f"{RUST}[WARNING]{RESET} Tokens file not found. Extracting from MIDI files...")

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

    print(f"{ORANGE}[INFO]{RESET} Total tokens extracted: {len(tokens)}")
    return tokens

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization as BatchNorm, Activation, Add

def inference_model(network_input, n_vocab):

    # Use the same model architecture as in training
    input_shape = (network_input.shape[1], network_input.shape[2])
    inputs = Input(shape=input_shape)
    x1 = LSTM(512, return_sequences=True, recurrent_dropout=0.3)(inputs)
    x2 = LSTM(512, return_sequences=True, recurrent_dropout=0.2)(x1)
    x = Add()([x1, x2]) 
    x3 = LSTM(384, recurrent_dropout=0.2)(x)
    x3 = BatchNorm()(x3)
    x3 = Dropout(0.4)(x3)
    fc1 = Dense(256)(x3)
    fc1 = Activation('relu')(fc1)
    fc1 = BatchNorm()(fc1)
    fc1 = Dropout(0.4)(fc1)
    x_proj = Dense(256)(x3) #added another residual connection
    x = Add()([x_proj, fc1])
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(n_vocab)(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam())
    
    print(f"{GOLD}[INFO]{RESET} Loading weights from checkpoint file...")
    model.load_weights('weights_checkpoint.keras')
    
    return model

def prepare_sequences(notes, pitches, n_vocab):
    print(f"{ORANGE}[INFO]{RESET} Preparing sequences for inference...")
    # Match sequence length with training (25)
    sequence_length = 30
    
    note_to_int = dict((note, number) for number, note in enumerate(pitches))

    network_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    n_patterns = len(network_input)
    
    # Reshape and normalize input
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_midi(prediction_output):
    print(f"{ORANGE}[INFO]{RESET} Converting output to MIDI...")
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        duration = 0.5  # default fallback
        if "_" in pattern:
            base, dur = pattern.rsplit("_", 1)
            try:
                duration = float(dur)
            except:
                duration = 0.5
        else:
            base = pattern

        if base.startswith("REST"):
            offset += duration
            continue

        elif '.' in base or base.isdigit():  # chord
            try:
                notes_in_chord = base.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                new_chord.quarterLength = duration
                output_notes.append(new_chord)
            except Exception as e:
                print(f"{RUST}[WARN]{RESET} Failed to parse chord: {pattern} ({e})")
        else:  # single note
            try:
                new_note = note.Note(base)
                new_note.offset = offset
                new_note.quarterLength = duration
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            except Exception as e:
                print(f"{RUST}[WARN]{RESET} Failed to parse note: {pattern} ({e})")

        offset += duration

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='generated.mid')
    print(f"{RUST}[INFO]{RESET} MIDI file created as 'generated.mid'")

def generate_notes(model, network_input, pitchnames, n_vocab):
    print(f"{ORANGE}[INFO]{RESET} Generating notes from model...")
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # Generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        
        top_idx = numpy.argsort(prediction[0])[-7:] 
        top_probabilities = prediction[0][top_idx]
        prob_norm = top_probabilities / numpy.sum(top_probabilities)
        selected_idx = numpy.random.choice(top_idx, p=prob_norm)
        result = int_to_note[selected_idx]
        prediction_output.append(result)

        pattern.append(selected_idx)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def generate():
    print(f"{MAROON}+=======================================+{RESET}")
    print(f"{MAROON}|  Starting Armstrong Inference Pipeline |{RESET}")
    print(f"{MAROON}+=======================================+{RESET}")
    
    notes = get_notes()
    pitches = sorted(set(item for item in notes))
    n_vocab = len(set(notes))
    
    print(f"{ORANGE}[INFO]{RESET} Vocabulary size: {n_vocab}")
    
    network_input, normalized_input = prepare_sequences(notes, pitches, n_vocab)
    model = inference_model(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitches, n_vocab)
    create_midi(prediction_output)
    print(f"{RUST}[INFO]{RESET} Generation complete!")

print(f"{ORANGE}Attempting to run Armstrong inference...{RESET} \n")
generate()