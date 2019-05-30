import py_midicsv as midi


def midi_to_csv(file):
    return midi.midi_to_csv(file)


csv = midi_to_csv("data/midi_files/bach_suite1_ii.mid")


for line in csv[:100]:
    print(line)
