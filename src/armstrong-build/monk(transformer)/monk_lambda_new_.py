import argparse
import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream, duration, analysis, key, scale, pitch
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding, Add, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from fractions import Fraction
import random

np.random.seed(42)

def get_note_rhythm_tokens():
    print("[INFO] Extracting pitch-duration tokens...")
    tokens = []
    keys_detected = []
    
    for file in glob.glob("./jazz_and_stuff/*.mid"):
        try:
            midi = converter.parse(file)
            
            key_analyzer = analysis.discrete.KeyAnalyzer(midi)
            curr_key = key_analyzer.process()
            if curr_key:
                keys_detected.append(curr_key.tonic.name + ' ' + curr_key.mode)
            
            try:
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:
                notes_to_parse = midi.flat.notes

            piece_tokens = []
            for element in notes_to_parse:
                dur = element.quarterLength
                if dur <= 0 or dur > 8:
                    continue
                    
                if isinstance(element, note.Note):
                    piece_tokens.append(f"{element.pitch}_{dur}")
                elif isinstance(element, chord.Chord):
                    chord_token = '.'.join(str(n) for n in element.normalOrder)
                    piece_tokens.append(f"{chord_token}_{dur}")
            
            tokens.extend(piece_tokens)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    print(f"[INFO] Detected {len(keys_detected)} key signatures")
    os.makedirs('./data', exist_ok=True)
    with open('./data/keys_detected', 'wb') as f:
        pickle.dump(keys_detected, f)
            
    with open('./data/tokens', 'wb') as f:
        pickle.dump(tokens, f)

    return tokens

def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

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

def transformer_block(x, embed_dim, num_heads, ff_dim, rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)(x, x)
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

def extract_pitch_class(note_token):
    if '_' not in note_token:
        return None
    
    pitch_part = note_token.split('_')[0]
    
    try:
        if '.' in pitch_part:  # Chord
            return [int(n) % 12 for n in pitch_part.split('.')]
        else:  # Note
            return [pitch.Pitch(pitch_part).midi % 12]
    except:
        return None

def extract_duration(note_token):
    if '_' not in note_token:
        return None
    
    try:
        return float(Fraction(note_token.split('_')[1]))
    except:
        return None

def analyze_rhythm_diversity(sequence):
    durations = [extract_duration(t) for t in sequence if extract_duration(t) is not None]
    if not durations:
        return 0
    
    unique_durations = len(set(durations))
    common_durations = {0.25, 0.5, 1.0, 1.5, 2.0}  
    meaningful_diversity = sum(1 for d in set(durations) if d in common_durations)
    
    pattern_score = 0
    for i in range(1, len(durations)):
        if durations[i] == durations[i-1]:
            pattern_score -= 0.1  
        if i >= 3 and durations[i] == durations[i-2] and durations[i-1] == durations[i-3]:
            pattern_score += 0.5  
    
    return (meaningful_diversity / max(1, len(common_durations))) + (pattern_score / max(1, len(durations)))

def analyze_melodic_contour(sequence):
    pitches = []
    for token in sequence:
        try:
            if '_' not in token:
                continue
                
            pitch_part = token.split('_')[0]
            if '.' not in pitch_part and not pitch_part.isdigit():
                pitches.append(pitch.Pitch(pitch_part).midi)
        except:
            continue
    
    if len(pitches) < 3:
        return 0
        
    directions = [1 if pitches[i] > pitches[i-1] else (-1 if pitches[i] < pitches[i-1] else 0) 
                  for i in range(1, len(pitches))]
    
    direction_changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
    
    steps = sum(1 for i in range(1, len(pitches)) if 1 <= abs(pitches[i] - pitches[i-1]) <= 2)
    small_leaps = sum(1 for i in range(1, len(pitches)) if 3 <= abs(pitches[i] - pitches[i-1]) <= 5)
    large_leaps = sum(1 for i in range(1, len(pitches)) if abs(pitches[i] - pitches[i-1]) > 5)
    
    step_ratio = steps / max(1, len(pitches) - 1)
    small_leap_ratio = small_leaps / max(1, len(pitches) - 1)
    large_leap_ratio = large_leaps / max(1, len(pitches) - 1)
    
    motion_score = (0.7 - abs(step_ratio - 0.65)) + (0.3 - abs(small_leap_ratio - 0.25)) + (0.1 - abs(large_leap_ratio - 0.1))
    
    return (direction_changes / max(1, len(directions))) + motion_score

def analyze_harmonic_consistency(sequence, key_name=None):
    if not key_name:
        curr_key = key.Key('C', 'major')
    else:
        parts = key_name.split()
        if len(parts) >= 2:
            curr_key = key.Key(parts[0], parts[1])
        else:
            curr_key = key.Key('C', 'major')
    
    scale_degrees = [n.pitch.midi % 12 for n in curr_key.getScale().getPitches()]
    
    in_key_count = 0
    total_notes = 0
    
    for token in sequence:
        pitch_classes = extract_pitch_class(token)
        if not pitch_classes:
            continue
            
        total_notes += len(pitch_classes)
        for pc in pitch_classes:
            if pc in scale_degrees:
                in_key_count += 1
    
    if total_notes == 0:
        return 0
        
    return in_key_count / total_notes

def reward_function(sequence, key_name=None):
    if not sequence:
        return 0
    
    base_reward = 0
    consonant_intervals = {0, 3, 4, 5, 7, 8, 9, 12}  # Unison, 3rds, 4th, 5th, 6ths, octave
    strong_consonances = {0, 7, 12}  # Unison, perfect 5th, octave
    
    for i in range(1, len(sequence)):
        try:
            prev_pitches = extract_pitch_class(sequence[i - 1])
            curr_pitches = extract_pitch_class(sequence[i])
            
            if not prev_pitches or not curr_pitches:
                continue
                
            interval = (curr_pitches[0] - prev_pitches[0]) % 12
            
            if interval in strong_consonances:
                base_reward += 1.5
            elif interval in consonant_intervals:
                base_reward += 1
            else:
                base_reward -= 0.5
        except Exception as e:
            continue
    
    if len(sequence) > 1:
        base_reward = base_reward / (len(sequence) - 1)
    
    rhythm_score = analyze_rhythm_diversity(sequence) * 2 
    
    melody_score = analyze_melodic_contour(sequence) * 3  
    
    harmonic_score = analyze_harmonic_consistency(sequence, key_name) * 2
    
    total_score = (base_reward * 0.3) + (rhythm_score * 0.3) + (melody_score * 0.2) + (harmonic_score * 0.2)
    
    return total_score

def top_k_sample(preds, k=5, temperature=0.8):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    # Get top k indices
    top_indices = preds.argsort()[-k:][::-1]
    top_probs = preds[top_indices]
    top_probs /= np.sum(top_probs)
    
    return np.random.choice(top_indices, p=top_probs)

def generate_with_rl(model, network_input, int_to_token, token_to_int, num_tokens=500, trials=8):
    key_name = None
    try:
        with open('./data/keys_detected', 'rb') as f:
            keys = pickle.load(f)
            if keys:
                key_name = random.choice(keys)
                print(f"[INFO] Using detected key: {key_name}")
    except:
        print("[INFO] No key information available, using default")
    
    best_output = []
    best_reward = float('-inf')

    for trial in range(trials):
        print(f"[INFO] Generation trial {trial+1}/{trials}")
        start = np.random.randint(0, len(network_input) - 1)
        pattern = list(network_input[start])
        prediction_output = []
        
        start_temp = 1.2
        end_temp = 0.7
        
        for i in range(num_tokens):
            temperature = start_temp - (i / num_tokens) * (start_temp - end_temp)
            
            prediction_input = np.reshape(pattern, (1, len(pattern)))
            prediction = model.predict(prediction_input, batch_size=1, verbose=0)[0]
            
            k = max(5, int(10 - (i / num_tokens) * 5))  #
            
            index = top_k_sample(prediction, k=k, temperature=temperature)
            result = int_to_token[index]
            prediction_output.append(result)
            pattern.append(index)
            pattern = pattern[1:]
        
        reward = reward_function(prediction_output, key_name)
        print(f"[INFO] Trial {trial+1} reward: {reward:.4f}")
        
        if reward > best_reward:
            best_reward = reward
            best_output = prediction_output
            
    print(f"[INFO] Best generation reward: {best_reward:.4f}")
    return best_output

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
        model = create_transformer_model(
            seq_len=network_input.shape[1], 
            vocab_size=n_vocab,
            embed_dim=256,
            num_heads=8,
            ff_dim=512,
            num_layers=3
        )
        model.summary()

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

    model = create_transformer_model(
        seq_len=network_input.shape[1], 
        vocab_size=n_vocab,
        embed_dim=256,
        num_heads=8,
        ff_dim=512,
        num_layers=3
    )

    weight_files = sorted(glob.glob("./weights/weights-improvement-*.hdf5"))
    if not weight_files:
        print("[ERROR] No weights found!")
        return
    
    model.load_weights(weight_files[-1])
    print(f"[INFO] Loaded weights from {weight_files[-1]}")

    os.makedirs('./outputs', exist_ok=True)

    for song_idx in range(5):
        output = generate_with_rl(model, network_input, int_to_token, token_to_int, num_tokens=500, trials=8)
        create_midi(output, song_idx)

def create_midi(prediction_output, idx):
    print(f"[INFO] Creating MIDI file for song {idx}...")
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '_' not in pattern:
            continue
            
        pitch_part, dur_part = pattern.split('_')
        
        try:
            dur = float(Fraction(dur_part))
            
            if dur <= 0 or dur > 8:
                continue
                
            if '.' in pitch_part or pitch_part.isdigit():
                notes_in_chord = pitch_part.split('.')
                notes_list = []
                
                for current_note in notes_in_chord:
                    try:
                        new_note = note.Note(int(current_note))
                        new_note.duration = duration.Duration(dur)
                        new_note.storedInstrument = instrument.Piano()
                        notes_list.append(new_note)
                    except:
                        continue
                        
                if notes_list:  
                    new_chord = chord.Chord(notes_list)
                    new_chord.offset = offset
                    output_notes.append(new_chord)
            else:
                try:
                    new_note = note.Note(pitch_part)
                    new_note.duration = duration.Duration(dur)
                    new_note.offset = offset
                    new_note.storedInstrument = instrument.Piano()
                    output_notes.append(new_note)
                except:
                    continue
                    
            offset += dur
            
        except Exception as e:
            print(f"Error processing token '{pattern}': {e}")
            continue

    key_name = None
    try:
        with open('./data/keys_detected', 'rb') as f:
            keys = pickle.load(f)
            if keys:
                key_name = random.choice(keys)
                print(f"[INFO] Using detected key for MIDI file: {key_name}")
                parts = key_name.split()
                if len(parts) >= 2:
                    ks = key.KeySignature(key.Key(parts[0], parts[1]).sharps)
                    ks.offset = 0.0
                    output_notes.insert(0, ks)
    except:
        print("[INFO] No key information available for MIDI file")

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=f'./outputs/transformer_generated_{idx}.mid')
    print(f"[INFO] Saved to ./outputs/transformer_generated_{idx}.mid")
    
   

def main():
    parser = argparse.ArgumentParser(description="Transformer-based Music Generator with Music Theory")
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