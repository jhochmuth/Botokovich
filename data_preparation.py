import os

import re

import music21

import numpy as np

import py_midicsv as midi

from tqdm import tqdm


BROKEN_FILES = {"bwv984.mid", "bwv986.mid"}

KEY_TRANSPOSITION_VALUES = {0: 0,
                            1: 5,
                            2: -2,
                            3: 3,
                            4: -4,
                            5: 1,
                            6: -6,
                            -1: -5,
                            -2: 2,
                            -3: -3,
                            -4: 4,
                            -5: -1,
                            -6: -3}

CHORALE_REGEX = re.compile("bwv(\d+).mid")


def midi_command_formatting(command):
    """Converts a string containing a midi command to an array."""
    command = command.strip("\n")
    command = command.split(", ")
    return command


def midi_to_csv(filename):
    """Extracts all commands from a midi file and converts them to array format."""
    csv = midi.midi_to_csv(filename)
    return [midi_command_formatting(command) for command in csv]


def get_transposition_value(midi_command_list):
    transposition_value = None

    for command in midi_command_list:
        if command[2] == "Key_signature":
            transposition_value = KEY_TRANSPOSITION_VALUES[int(command[3])]
            break

    if not transposition_value:
        transposition_value = 0

    return transposition_value


def extract_note_values(midi_command_list):
    transposition_value = get_transposition_value(midi_command_list)
    return [int(command[4]) + transposition_value for command in midi_command_list
            if command[2] == "Note_on_c" and int(command[5]) > 0 and int(command[3]) == 0]


def extract_notes_from_file(filename):
    midi_command_list = midi_to_csv(filename)
    return extract_note_values(midi_command_list)


def extract_notes_from_all_files(dir):
    sequences = list()
    for filename in os.listdir(dir):
        if filename.endswith(".mid"):
            sequences.append(extract_notes_from_file(os.path.join(dir, filename)))
    np.save("data/cello_pieces", sequences)


def extract_simplified_chord_encoding(filename):
    midi_command_list = midi_to_csv(filename)

    pulse = int(midi_command_list[0][5])
    transposition_value = get_transposition_value(midi_command_list)
    chords_time = [list() for _ in range(400)]
    for command in midi_command_list:
        if command[2] == "Note_on_c":
            if int(command[1]) % (pulse // 4) == 0:
                ind = int(command[1]) // (pulse // 4)
                if ind < 400:
                    chords_time[ind].append(int(command[4]) + transposition_value)

    for timestep in chords_time:
        delete_notes = list()
        for note in timestep:
            if note < 36:
                delete_notes.append(note)
            elif note > 84:
                delete_notes.append(note)
        for note in delete_notes:
            timestep.remove(note)

    current_notes = list()
    temp = [set() for _ in range(400)]
    for t in range(400):
        stop_notes = list()
        new_notes = list()
        changes = chords_time[t]
        for note in changes:
            if note in current_notes:
                current_notes.remove(note)
                stop_notes.append(note)
            else:
                new_notes.append(note)
        for note in stop_notes:
            changes.remove(note)
        for note in current_notes:
            temp[t].add((note, 2))
        for note in new_notes:
            temp[t].add((note, 1))
            current_notes.append(note)
    chords_time = temp

    temp = list()
    for chord in chords_time:
        current_chord = [0 for _ in range(49)]
        for note in chord:
            note_index = note[0] - 36
            current_chord[note_index] = note[1]
        temp.append(current_chord)
    chords_time = temp

    for i, _ in enumerate(chords_time):
        chords_time[i] = list(map(str, chords_time[i]))
        chords_time[i] = "".join(chords_time[i])
    chords_time = " ".join(chords_time)

    return chords_time


def extract_chord_encodingv2(filename, steps_per_quarter=12):
    """Uses music21 library to extract chords at each timestep."""
    stream = music21.midi.translate.midiFilePathToStream(filename)
    time_steps = [list() for _ in range(int(stream.duration.quarterLength * steps_per_quarter))]

    for element in stream.recurse(classFilter=('Chord', 'Note')):
        time = int(element.offset * steps_per_quarter)
        duration_steps = int(element.duration.quarterLength * steps_per_quarter)

        if isinstance(element, music21.note.Note):
            for duration in range(duration_steps):
                if time + duration > len(time_steps) - 1:
                    break
                elif element.pitch.ps < 36 or element.pitch.ps > 96:
                    continue
                elif duration == 0:
                    time_steps[time + duration].append((int(element.pitch.ps), 1))
                else:
                    time_steps[time + duration].append((int(element.pitch.ps), 2))

        elif isinstance(element, music21.chord.Chord):
            for duration in range(duration_steps):
                if time + duration > len(time_steps) - 1:
                    break
                for note in element.pitches:
                    if note.ps < 36 or note.ps > 96:
                        continue
                    elif duration == 0:
                        time_steps[time + duration].append((int(note.ps), 1))
                    else:
                        time_steps[time + duration].append((int(note.ps), 2))

    # Transpose all sequences to C major. Music21 does not seem to automatically find key signature.
    commands = midi_to_csv(filename)
    transposition_value = get_transposition_value(commands)
    if transposition_value != 0:
        for i, step in enumerate(time_steps):
            transposed_step = list()
            for note in step:
                transposed_pitch = note[0] + transposition_value
                if 36 <= transposed_pitch <= 96:
                    transposed_step.append((note[0] + transposition_value, note[1]))
            time_steps[i] = transposed_step

    chord_sequence = list()
    for step in time_steps:
        chord = ["0" for _ in range(61)]
        for note in step:
            chord[note[0] - 36] = str(note[1])
        chord_sequence.append("".join(chord))
    chord_sequence = " ".join(chord_sequence)
    return chord_sequence


# TODO: Check the effects of repeated notes. Important for ensemble pieces.
# TODO: Find out what is causing music21 library to not load offsets correctly.
def extract_note_encoding(filename):
    """Function to extract notewise encoding. This version specifies when notes are stopped."""
    chord_sequence = extract_chord_encodingv2(filename)
    chord_sequence = chord_sequence.split(" ")
    note_sequence = ""

    current_notes = set()
    for chord in chord_sequence:
        stopped_notes = set()
        for note_index in current_notes:
            if chord[note_index] == "0":
                stopped_notes.add(note_index)
                note_sequence = "{} stop{}".format(note_sequence, str(note_index))
        for note_index in stopped_notes:
            current_notes.remove(note_index)
        for i, char in enumerate(chord):
            if char == "1":
                note_sequence = "{} {}".format(note_sequence, str(i))
                current_notes.add(i)
        note_sequence = "{} step".format(note_sequence)

    return note_sequence


def midi_selection(filename, selection="all"):
    if not filename.endswith(".mid") or filename in BROKEN_FILES:
        return False

    if selection == "all":
        return True
    elif selection == "bach":
        return "bwv" in filename
    elif selection == "chorale":
        m = CHORALE_REGEX.match(filename)
        return 250 <= int(m.group(1)) <= 438
    elif selection == "haydn":
        return "haydn" in filename
    elif selection == "mozart":
        return "mz" in filename
    else:
        raise Exception


def extract_sequences_from_all_files(dir, output_file, extraction_method, selection="all", debug=False):
    sequences = list()

    for filename in tqdm(os.listdir(dir)):
        if midi_selection(filename, selection):
            if debug:
                print("Extracting sequences from: {}".format(filename))
            sequences.append(extraction_method(os.path.join(dir, filename)))

    print("Saving file to: {}".format(output_file))
    np.save(output_file, sequences)


def main():
    #extract_notes_from_all_files("data/train_midi_files/bach_cello_suites")
    #extract_sequences_from_all_files("data/train_midi_files/major", "data/chord_sequences", extract_simplified_chord_encoding)
    #extract_sequences_from_all_files("data/train_midi_files/major", "data/chord_sequences", extract_chord_encodingv2)

    extract_sequences_from_all_files("data/train_midi_files/major", "data/note_sequences", extract_note_encoding)


if __name__ == "__main__":
    main()
