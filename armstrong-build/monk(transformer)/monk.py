import glob
import pickle
import numpy as np
import os
from music21 import converter, instrument, note, chord, stream
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ANSI escape color codes - Fall colors palette
ORANGE = '\033[38;5;208m'  # Bright orange
RUST = '\033[38;5;166m'    # Rust/deep orange
BROWN = '\033[38;5;130m'   # Brown
GOLD = '\033[38;5;220m'    # Gold/amber
MAROON = '\033[38;5;88m'   # Deep maroon/burgundy
RESET = '\033[0m'          # Reset to default
BOLD = '\033[1m'           # Bold text

def get_notes(directory="./jazz_and_stuff", train=True):
    """Extract notes and chords with durations from MIDI files"""
    print(f"{ORANGE}[INFO]{RESET} Extracting notes and chords with durations from MIDI files...")
    tokens = []

    # For inference, just load existing tokens if available
    if not train:
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

def prepare_sequences(notes, n_vocab, sequence_length=30, train=True):
    """Prepare the sequences for training or inference"""
    print(f"{ORANGE}[INFO]{RESET} Preparing sequences...")
    
    pitches = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitches))
    int_to_note = dict((number, note) for number, note in enumerate(pitches))
    
    if train:
        net_in = []
        net_out = []
        
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            net_in.append([note_to_int[char] for char in sequence_in])
            net_out.append(note_to_int[sequence_out])
        
        n_patterns = len(net_in)
        
        # Reshape for transformer input [batch_size, seq_length]
        net_in = np.array(net_in)
        net_out = tf.keras.utils.to_categorical(net_out, num_classes=n_vocab)
        
        print(f"{ORANGE}[INFO]{RESET} Total patterns: {n_patterns}")
        return net_in, net_out, note_to_int, int_to_note
    else:
        network_input = []
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
        
        return np.array(network_input), note_to_int, int_to_note

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    """Create a transformer encoder block"""
    # Multi-head attention
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    
    # Add & Norm (Add the input and normalize)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed Forward
    ffn_output = Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    
    # Add & Norm (Add the attention output and normalize)
    sequence_output = LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
    
    return sequence_output

def build_transformer_model(vocab_size, sequence_length=30, embed_dim=256):
    """Build the transformer model architecture"""
    print(f"{ORANGE}[INFO]{RESET} Creating transformer model")
    
    # Input layer
    inputs = Input(shape=(sequence_length,))
    
    # Embedding layer
    embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_dim)(inputs)
    
    # Positional encoding
    position_indices = tf.range(sequence_length, dtype=tf.float32)[tf.newaxis, :]
    pos_encoding = positional_encoding(sequence_length, embed_dim)
    x = embedding_layer + pos_encoding
    
    # Transformer encoder blocks
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=512, dropout=0.2)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=512, dropout=0.2)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=512, dropout=0.2)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(vocab_size, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    
    return model

def positional_encoding(position, d_model):
    """Create positional encoding for transformer model"""
    angles = 1 / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    position_matrix = np.arange(position)[:, np.newaxis]
    
    angle_rads = position_matrix * angles
    
    # Apply sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def train_transformer(model, network_input, network_output, finetune=False):
    """Train the transformer model"""
    print(f"{ORANGE}[INFO]{RESET} Starting training...")
    filepath = "weights_transformer_checkpoint.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    if finetune:
        print(f"{GOLD}[INFO]{RESET} Loading weights for fine-tuning...")
        model.load_weights('weights_transformer_checkpoint.keras')
    
    history = model.fit(
        network_input, 
        network_output, 
        epochs=350,  # Reduced from 550 as transformers typically converge faster
        batch_size=64,  # Smaller batch size to accommodate larger model
        callbacks=callbacks_list
    )
    
    final_loss = history.history['loss'][-1]
    with open('./transformer_loss.txt', 'w') as file:
        file.write(str(final_loss))
    
    print(f"{ORANGE}[INFO]{RESET} Final loss value ({final_loss}) saved to 'transformer_loss.txt'.")
    print(f"{RUST}[INFO]{RESET} Training complete.")

def generate_notes(model, seed_input, int_to_note, n_vocab, num_notes=500, temperature=1.0):
    """Generate new notes using the trained transformer model"""
    print(f"{ORANGE}[INFO]{RESET} Generating notes from transformer model...")
    
    # Start with a random seed
    current_pattern = seed_input
    prediction_output = []
    
    # Generate notes one by one
    for _ in range(num_notes):
        # Predict the next note
        prediction = model.predict(np.array([current_pattern]), verbose=0)[0]
        
        # Apply temperature scaling for more creative outputs
        prediction = np.log(prediction) / temperature
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        
        # Sample from the distribution with weighted probabilities
        top_indices = np.argsort(prediction)[-10:]  # Get top 10 predictions
        top_probs = prediction[top_indices]
        top_probs = top_probs / np.sum(top_probs)  # Normalize to sum to 1
        
        # Weighted random choice from top candidates
        next_index = np.random.choice(top_indices, p=top_probs)
        result = int_to_note[next_index]
        prediction_output.append(result)
        
        # Update pattern with the generated note
        current_pattern = np.append(current_pattern[1:], next_index)
    
    return prediction_output

def create_midi(prediction_output):
    """Convert the predicted output to a MIDI file"""
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
    midi_stream.write('midi', fp='transformer_generated.mid')
    print(f"{RUST}[INFO]{RESET} MIDI file created as 'transformer_generated.mid'")

def train_network():
    """Main function for training the transformer model"""
    print(f"{MAROON}+==========================================+{RESET}")
    print(f"{MAROON}|  Starting Transformer Training Pipeline  |{RESET}")
    print(f"{MAROON}+==========================================+{RESET}")
    
    notes = get_notes()
    n_vocab = len(set(notes))
    print(f"{ORANGE}[INFO]{RESET} Vocabulary size: {n_vocab}")
    
    network_input, network_output, note_to_int, int_to_note = prepare_sequences(notes, n_vocab, train=True)
    
    model = build_transformer_model(n_vocab)
    print(f"{ORANGE}[INFO]{RESET} Model summary:")
    model.summary()
    
    train_transformer(model, network_input, network_output, finetune=False)
    
    # Save note mappings for inference
    mapping_data = {
        'note_to_int': note_to_int,
        'int_to_note': int_to_note
    }
    with open('./data/note_mappings.pkl', 'wb') as f:
        pickle.dump(mapping_data, f)
    print(f"{ORANGE}[INFO]{RESET} Note mappings saved to './data/note_mappings.pkl'")

def generate():
    """Main function for generating music with the transformer model"""
    print(f"{MAROON}+============================================+{RESET}")
    print(f"{MAROON}|  Starting Transformer Inference Pipeline   |{RESET}")
    print(f"{MAROON}+============================================+{RESET}")
    
    notes = get_notes(train=False)
    n_vocab = len(set(notes))
    print(f"{ORANGE}[INFO]{RESET} Vocabulary size: {n_vocab}")
    
    # Load the note mappings
    try:
        with open('./data/note_mappings.pkl', 'rb') as f:
            mapping_data = pickle.load(f)
            note_to_int = mapping_data['note_to_int']
            int_to_note = mapping_data['int_to_note']
    except FileNotFoundError:
        print(f"{RUST}[WARNING]{RESET} Note mappings not found. Creating new mappings...")
        network_input, note_to_int, int_to_note = prepare_sequences(notes, n_vocab, train=False)
    
    # Load the model
    model = build_transformer_model(n_vocab)
    model.load_weights('weights_transformer_checkpoint.keras')
    
    # Generate a random seed
    seed_index = np.random.randint(0, len(notes) - 31)
    seed_sequence = notes[seed_index:seed_index + 30]
    seed_pattern = [note_to_int[note] for note in seed_sequence]
    
    # Generate music
    prediction_output = generate_notes(model, np.array(seed_pattern), int_to_note, n_vocab, temperature=1.2)
    create_midi(prediction_output)
    print(f"{RUST}[INFO]{RESET} Generation complete!")

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Music Generation with Transformer')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'],
                        help='Whether to train the model or generate music')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"{ORANGE}Attempting to train Transformer Music Model...{RESET} \n")
        train_network()
        print(f"{RUST}Transformer training complete.{RESET}")
    else:
        print(f"{ORANGE}Attempting to run Transformer inference...{RESET} \n")
        generate()