import pickle
import pretty_midi

# file_path = "/home/azuma/workspace/dataset/chord_lstm_data/indexroll/A/A/A/TRAAAGR128F425B14B/1d9d16a9da90c090809c153754823c2b.pickle"
# file_path = "/home/azuma/workspace/dataset/chord_lstm_data/indexroll/A/A/A/TRAAAGR128F425B14B/b97c529ab9ef783a849b896816001748.pickle"
file_path = "/home/azuma/workspace/dataset/chord_lstm_data/indexroll/A/A/A/TRAAAZF12903CCCF6B/10288ea8e07b70c17f872fda82b94330.pickle"
with open(file_path, 'rb') as f:
    indexroll = pickle.load(f)

# print(indexroll)
speed = 0.15
pm = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=300)
instrument = pretty_midi.Instrument(0)

for i in range(len(indexroll)):
    if indexroll[i] == -1:
        continue
    else:
        note_number = indexroll[i]
    note = pretty_midi.Note(velocity=100, pitch=note_number, start=speed*i, end=speed*(i+1))
    instrument.notes.append(note)
    # for note_number in indexroll[i]:
    #     note = pretty_midi.Note(velocity=100, pitch=note_number, start=speed*i, end=speed*(i+1))
    #     instrument.notes.append(note)

pm.instruments.append(instrument)
pm.write('test.mid')