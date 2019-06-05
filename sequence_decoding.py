"""Provides the functionality to convert encoded sequences into midi files.

"""
import music21

import pandas as pd

import pygame

from tqdm import tqdm


def load_sequences_from_file(filename):
    sequences = pd.read_csv(filename)
    sequences = list(sequences["0"])
    return sequences


def create_note(pitch, offset, duration):
    duration = music21.duration.Duration(duration)
    note = music21.note.Note(pitch, duration=duration)
    note.offset = offset
    return note


def create_midi_file(sequence, output_filename):
    sequence = sequence.split(" ")
    notes = list()
    current_notes = list()
    steps = 0
    step_size = 1 / 12
    max_length = 1

    for element in sequence:
        if element == "xxbos" or element == "":
            continue

        elif element == "step":
            steps += 1
            delete_notes = list()
            for i, note in enumerate(current_notes):
                note[2] += 1
                if note[2] >= max_length / step_size:
                    new_note = create_note(note[0] + 36, note[1] * step_size, note[2] * step_size)
                    notes.append(new_note)
                    delete_notes.append(note)
            for note in delete_notes:
                current_notes.remove(note)

        elif "stop" in element:
            pitch = int(element[4:])
            current_pitches = [n[0] for n in current_notes]
            if pitch in current_pitches:
                new_note = create_note(note[0] + 36, note[1] * step_size, note[2] * step_size)
                notes.append(new_note)
                index = current_pitches.index(pitch)
                del current_notes[index]

        else:
            pitch = int(element)
            offset_steps = steps
            duration_steps = step_size
            current_notes.append([pitch, offset_steps, duration_steps])

    for note in current_notes:
        new_note = create_note(note[0] + 36, note[1] * step_size, note[2] * step_size)
        notes.append(new_note)

    piano = music21.instrument.fromString("Piano")
    notes.insert(0, piano)
    stream = music21.stream.Stream(notes)

    output_path = "data/generated_midi_files/{}".format(output_filename)
    stream.write(fmt="midi", fp=output_path)


def convert_all_sequences_in_file(filename):
    sequences = load_sequences_from_file(filename)
    for i, sequence in tqdm(enumerate(sequences)):
        create_midi_file(sequence, "generated_{}.mid".format(i))


def play_midi_file(filename):
    pygame.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.wait(1000)


convert_all_sequences_in_file("data/generated_sequences/sequences_300hs.csv")
play_midi_file("data/generated_midi_files/generated_4.mid")
