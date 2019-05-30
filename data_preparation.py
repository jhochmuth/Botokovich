import os

import py_midicsv as midi


def midi_command_formatting(command):
    command = command.strip("\n")
    command = command.split(", ")
    return command


def midi_to_csv(filename):
    csv = midi.midi_to_csv(filename)
    return [midi_command_formatting(command) for command in csv]


def extract_note_values(midi_command_list):
    return [command[4] for command in midi_command_list
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


pieces = extract_notes_from_all_files("data/midi_files")
