import py_midicsv as midi


def midi_command_formatting(command):
    command = command.strip("\n")
    command = command.split(", ")
    return command


def midi_to_csv(file):
    csv = midi.midi_to_csv(file)
    return [midi_command_formatting(command) for command in csv]


# TODO: Replace list comprehension with normal loop so that you can break after finishing with notes.
def extract_note_values(midi_command_list):
    return [command[4] for command in midi_command_list if command[2] == "Note_on_c"]


midi_command_list = midi_to_csv("data/midi_files/bach_suite1_i.mid")
print(midi_command_list)
note_list = extract_note_values(midi_command_list)
print(note_list)
