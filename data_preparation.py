import os

import numpy as np

import py_midicsv as midi


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
    key_transposition_values = {0: 0,
                                1: 5,
                                2: -2,
                                3: 3,
                                -1: -5,
                                -2: 2,
                                -3: -3}

    for command in midi_command_list:
        if command[2] == "Key_signature":
            transposition_value = key_transposition_values[int(command[3])]
            break

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
    np.save("pieces", sequences)


def extract_simplified_chord_encoding(midi_command_list):
    transposition_value = get_transposition_value(midi_command_list)
    chords_time = [list() for _ in range(400)]
    for command in midi_command_list:
        if command[2] == "Note_on_c":
            if int(command[1]) % 120 == 0:
                ind = int(command[1]) // 120
                if ind < 400:
                    chords_time[ind].append(int(command[4]) + transposition_value)

    # 36-84
    # TODO: Change so that notes are modified instead of deleted.
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
        current_chord = [0 for _ in range(48)]
        for note in chord:
            note_index = note[0] - 36
            current_chord[note_index] = note[1]
        temp.append(current_chord)
    chords_time = temp

    for i in range(len(chords_time)):
        chords_time[i] = list(map(str, chords_time[i]))
        chords_time[i] = "".join(chords_time[i])
    chords_time = " ".join(chords_time)

    return chords_time


def extract_chords_from_all_files(dir):
    sequences = list()
    for filename in os.listdir(dir):
        if filename.endswith(".mid"):
            commands = midi_to_csv(os.path.join(dir, filename))
            sequences.append(extract_simplified_chord_encoding(commands))
    np.save("chord_sequences", sequences)



def main():
    #extract_notes_from_all_files("data/midi_files")
    extract_chords_from_all_files("data/midi_files")


if __name__ == "__main__":
    main()
