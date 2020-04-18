import pickle
import pretty_midi

file_path = "/home/azuma/workspace/dataset/chord_lstm_data/indexroll/A/A/A/TRAAAGR128F425B14B/dac3cdd0db6341d8dc14641e44ed0d44.pickle"
# file_path = "/home/azuma/workspace/dataset/chord_lstm_data/indexroll/A/A/A/TRAAAZF12903CCCF6B/10288ea8e07b70c17f872fda82b94330.pickle"
with open(file_path, 'rb') as f:
    indexroll = pickle.load(f)

print(indexroll)
speed = 0.15
pm = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=300)
instrument = pretty_midi.Instrument(0)

pre_note = 0
counter = 1
for i in range(len(indexroll)):
    if indexroll[i] == -1:
        note_number = 0
    else:
        note_number = indexroll[i]

    if pre_note == note_number:
        counter += 1
    else:
        if note_number != 0:
            note = pretty_midi.Note(velocity=100,
                                    pitch=note_number,
                                    start=speed*(i+1-counter),
                                    end=speed*(i+1))
            instrument.notes.append(note)
        counter = 1
    pre_note = note_number

pm.instruments.append(instrument)
pm.write('test.mid')