import os

import music21

import numpy as np

import py_midicsv as midi


# TODO: Investigate if sonat-10 is truly broken.
BROKEN_FILES = {"data/midi_files/piano/sonat-10.mid"}

KEY_TRANSPOSITION_VALUES = {0: 0,
                            1: 5,
                            2: -2,
                            3: 3,
                            4: -4,
                            5: 1,
                            -1: -5,
                            -2: 2,
                            -3: -3,
                            -4: 4,
                            -5: -1}


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


def extract_simplified_chord_encodingv2(filename):
    print("Extracting chords from: {}".format(filename))
    time_steps = [list() for _ in range(200 * 12)]
    stream = music21.midi.translate.midiFilePathToStream(filename)

    for element in stream.recurse(classFilter=('Chord', 'Note')):
        time = int(element.offset * 12)
        duration_steps = int(element.duration.quarterLength * 12)

        if isinstance(element, music21.note.Note):
            for duration in range(duration_steps):
                if time + duration > len(time_steps) - 1:
                    break
                elif element.pitch.ps > 84 or element.pitch.ps < 32:
                    pass
                elif duration == 0:
                    time_steps[time + duration].append((int(element.pitch.ps), 1))
                else:
                    time_steps[time + duration].append((int(element.pitch.ps), 2))

        elif isinstance(element, music21.chord.Chord):
            for duration in range(duration_steps):
                if time + duration > len(time_steps) - 1:
                    break
                for note in element.pitches:
                    if note.ps > 84 or note.ps < 36:
                        pass
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
                transposed_step.append((note[0] + transposition_value, note[1]))
            time_steps[i] = transposed_step

    chord_sequence = list()
    for step in time_steps:
        chord = ["0" for _ in range(52)]
        for note in step:
            chord[note[0] - 36] = str(note[1])
        chord_sequence.append("".join(chord))
    chord_sequence = " ".join(chord_sequence)
    return chord_sequence


def extract_chords_from_all_files(dir, extraction_method):
    sequences = list()
    for filename in os.listdir(dir):
        if filename.endswith(".mid") and filename not in BROKEN_FILES:
            sequences.append(extraction_method(os.path.join(dir, filename)))
    np.save("data/chord_sequences", sequences)


def main():
    #extract_notes_from_all_files("data/midi_files/bach_cello_suites")
    #extract_chords_from_all_files("data/midi_files/piano", extract_simplified_chord_encoding)
    extract_chords_from_all_files("data/midi_files/piano", extract_simplified_chord_encodingv2)


if __name__ == "__main__":
    main()
