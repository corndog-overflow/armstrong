import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Activation, BatchNormalization as BatchNorm, Add
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

# ANSI escape color codes
ORANGE = '\033[38;5;208m'
RUST = '\033[38;5;166m'
BROWN = '\033[38;5;130m'
GOLD = '\033[38;5;220m'
MAROON = '\033[38;5;88m'
RESET = '\033[0m'
BOLD = '\033[1m'

def get_notes(directory="./jazz_and_stuff"):
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
            duration = round(element.quarterLength * 2) / 2

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
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as filepath:
        pickle.dump(tokens, filepath)

    print(f"{ORANGE}[INFO]{RESET} Tokens saved to './data/tokens'")
    return tokens

def prepare_sequences_train(notes, n_vocab):
    print(f"{ORANGE}[INFO]{RESET} Preparing input/output sequences...")
    sequence_length = 30

    pitches = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitches))

    net_in = []
    net_out = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        net_in.append([note_to_int[char] for char in sequence_in])
        net_out.append(note_to_int[sequence_out])

    n_patterns = len(net_in)

    net_in = numpy.reshape(net_in, (n_patterns, sequence_length, 1)) / float(n_vocab)
    net_out = tf.keras.utils.to_categorical(net_out, num_classes=n_vocab)

    print(f"{ORANGE}[INFO]{RESET} Total patterns: {n_patterns}")
    return net_in, net_out, note_to_int, dict((v, k) for k, v in note_to_int.items())

def layer_model(network_input, n_vocab):
    print(f"{ORANGE}[INFO]{RESET} creating armstrong model")
    inputs = Input(shape=(network_input.shape[1], network_input.shape[2]))

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
    x_proj = Dense(256)(x3)
    x = Add()([x_proj, dense1])
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(n_vocab)(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(clipnorm=1.0))
    return model

def train(model, network_input, network_output, finetune=True):
    print(f"{ORANGE}[INFO]{RESET} Starting training...")
    filepath = "weights.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    if finetune and os.path.exists(filepath):
        print(f"{GOLD}[INFO]{RESET} Loading weights for fine-tuning...")
        model.load_weights(filepath)

    history = model.fit(network_input, network_output, epochs=550, batch_size=128, callbacks=[checkpoint])
    final_loss = history.history['loss'][-1]
    with open('./loss.txt', 'w') as file:
        file.write(str(final_loss))

    print(f"{ORANGE}[INFO]{RESET} Final loss value ({final_loss}) saved to 'loss.txt'.")
    print(f"{RUST}[INFO]{RESET} Training complete.")

def reward_function(sequence):
    reward = 0.0
    for i in range(1, len(sequence)):
        try:
            prev = sequence[i - 1].split("_")[0]
            curr = sequence[i].split("_")[0]
            if "." not in prev and "." not in curr:
                interval = abs(note.Note(prev).pitch.midi - note.Note(curr).pitch.midi)
                reward += 0.4 if interval <= 2 else (0.2 if interval <= 5 else -0.2)
        except:
            continue

    jazz_chords = {"0.4.7", "2.5.9", "4.7.11", "5.9.0", "7.10.2"}
    for token in sequence:
        if token.split("_")[0] in jazz_chords:
            reward += 0.5

    return reward / len(sequence)

def generate_sequence(model, network_input, int_to_note, n_vocab, sequence_length=30, gen_length=100):
    start = numpy.random.randint(0, len(network_input) - 1)
    pattern = list(network_input[start].flatten())
    generated = []
    for _ in range(gen_length):
        input_seq = numpy.reshape(pattern, (1, sequence_length, 1)) / float(n_vocab)
        prediction = model.predict(input_seq, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_note[index]
        generated.append(result)
        pattern.append(index)
        pattern = pattern[1:]
    return generated

def reinforce_update(model, sequences, rewards, note_to_int, n_vocab):
    optimizer = Adam(learning_rate=1e-4)
    sequence_length = 30
    with tf.GradientTape() as tape:
        loss = 0.0
        for seq, reward in zip(sequences, rewards):
            x_seq = [note_to_int[n] for n in seq[:sequence_length]]
            y_val = note_to_int[seq[sequence_length]]
            x = numpy.reshape(x_seq, (1, sequence_length, 1)) / float(n_vocab)
            y = tf.keras.utils.to_categorical([y_val], num_classes=n_vocab)
            pred = model(x, training=True)
            log_prob = tf.math.log(tf.reduce_sum(pred * y))
            loss -= log_prob * reward
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return float(loss)

def rl_fine_tune(model, network_input, note_to_int, int_to_note, n_vocab, episodes=50):
    print(f"{GOLD}[RL]{RESET} Starting reinforcement fine-tuning...")
    for ep in range(episodes):
        sequences, rewards = [], []
        for _ in range(8):
            seq = generate_sequence(model, network_input, int_to_note, n_vocab)
            reward = reward_function(seq)
            sequences.append(seq[:31])
            rewards.append(reward)
        loss = reinforce_update(model, sequences, rewards, note_to_int, n_vocab)
        print(f"{GOLD}[RL]{RESET} Episode {ep+1}/{episodes} | Avg reward: {numpy.mean(rewards):.4f} | Loss: {loss:.4f}")

def train_network():
    print(f"{MAROON}+=======================================+{RESET}")
    print(f"{MAROON}|  Starting Armstrong Training Pipeline |{RESET}")
    print(f"{MAROON}+=======================================+{RESET}")
    notes = get_notes()
    n_vocab = len(set(notes))
    print(f"{ORANGE}[INFO]{RESET} Vocabulary size: {n_vocab}")
    network_input, network_output, note_to_int, int_to_note = prepare_sequences_train(notes, n_vocab)
    model = layer_model(network_input, n_vocab)
    model.summary()
    print(f"{ORANGE}[INFO]{RESET} Total trainable parameters: {model.count_params()}")
    train(model, network_input, network_output, True)
    rl_fine_tune(model, network_input, note_to_int, int_to_note, n_vocab, episodes=50)

print(f"{ORANGE}Attempting to train Armstrong...{RESET} \n")
train_network()
print(f"{RUST}Armstrong training complete.{RESET}")
