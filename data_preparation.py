import os

import py_midicsv as midi


def midi_command_formatting(command):
    command = command.strip("\n")
    command = command.split(", ")
    return command


def midi_to_csv(filename):
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


def extract_notes_from_all_files(directory):
    pieces = list()
    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            pieces.append(extract_notes_from_file(os.path.join(directory, filename)))
    return pieces
