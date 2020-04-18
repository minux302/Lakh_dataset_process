import pickle
from collections import Counter
from pathlib import Path

import mido
import numpy as np
import pretty_midi as pm
from tqdm import tqdm


OCTAVE = 12
FS = 4  # Sampling frequency of the columns, i.e. each column is spaced apart by 1./fs seconds.
BAR_LEN = FS * 2


def is_containing_data_directly(data_dir):
    """Determine if data.suffix is directly under the data_dir by
       considering the structure of a lakh dataset.
       ex. lmd_matched/R/F/C/TRRFCAT128F426A5B1/data.mid
       If data_dir is 'TRRFCAT128F426A5B1', return True.
    """
    # tmp rule
    return data_dir.is_dir() and len(data_dir.name) != 1


def process_dir(original_dir, target_dir,
                original_suffix, target_suffix,
                process_file, **kwargs):
    for original_file in original_dir.glob('*.' + original_suffix):
        process_file(original_file,
                     target_dir /
                     original_file.name.replace(original_suffix,
                                                target_suffix),
                     **kwargs)


def process_lakh_dataset(original_root_dir, target_root_dir,
                         original_suffix, target_suffix,
                         process_file, **kwargs):
    """Create a proccesed dataset from original dataset (Lakh dataset format)
       Lakh dataset: https://colinraffel.com/projects/lmd/
        Args:
            original_root_dir (pathlib object): Path to root dir of Lakh dir
                format, like `lmd_matched`.
            target_root_dir (pathlib objext): Path to root dir of Lakh dir
                format. If it does not exists, it will be created.
            original_suffix (str): Each data suffix before it is processed.
            target_suffix (str): Each ata suffix after it is processed.
            process_file (func): Func for processing each data.
            (**kwargs: Optional args for process_file())
    """
    for original_dir in tqdm(original_root_dir.glob('**/*')):
        if is_containing_data_directly(original_dir):
            target_dir = Path(str(original_dir).replace(original_root_dir.name,
                                                        target_root_dir.name))
            if not(target_dir.exists()):
                target_dir.mkdir(parents=True)
            process_dir(original_dir, target_dir,
                        original_suffix, target_suffix,
                        process_file, **kwargs)


def change_tempo_midi(midi_file, save_path):
    try:
        mid = mido.MidiFile(midi_file)
        new_mid = mido.MidiFile()

        new_mid.ticks_per_beat = mid.ticks_per_beat
        for track in mid.tracks:
            new_track = mido.MidiTrack()
            for msg in track:
                new_msg = msg.copy()
                if new_msg.type == 'set_tempo':
                    new_msg.tempo = 500000
                new_track.append(new_msg)
            new_mid.tracks.append(new_track)
        new_mid.save(save_path)

    except (KeyError, OSError, EOFError, ValueError,
            AttributeError, IndexError, ZeroDivisionError) as e:
        print(str(midi_file), e)


def pianoroll_to_histo(pianoroll):
    bar_num = pianoroll.shape[1] // BAR_LEN
    histo_over_octave = np.zeros((pianoroll.shape[0], bar_num))
    for i in range(bar_num):
        histo_over_octave[:, i] = np.sum(pianoroll[:, i*BAR_LEN:(i+1)*BAR_LEN],
                                         axis=1)
    return histo_over_octave


def compress_octave_notes(histo_over_octave):
    histo = np.zeros((OCTAVE, histo_over_octave.shape[1]))
    octave_num = histo_over_octave.shape[0] // OCTAVE
    for i in range(octave_num - 1):
        histo = np.add(histo, histo_over_octave[i*OCTAVE:(i+1)*OCTAVE])
    return histo


def midi_to_histo(midi_file, save_path):
    """Generate histogram (histo) from midi_file.
        histo (np.array):
            histo[i][j]: non zero num per bar_j for key_i.
            key_i is 0-11, it means C, Db, D, ..., Bb, B.
            bar_j is index in range(time length in song // time length in bar).
    """
    try:
        mid = pm.PrettyMIDI(str(midi_file))
        pianoroll = mid.get_piano_roll()
    except (KeyError, OSError, EOFError, ValueError,
            AttributeError, IndexError, ZeroDivisionError) as e:
        print(str(midi_file), e)
        return

    histo_over_octave = pianoroll_to_histo(pianoroll)  # shape: (128, pianoroll.shape[1] // bar_len)
    histo = compress_octave_notes(histo_over_octave)  # shape: (12, pianoroll.shape[1] // bar_len)
    pickle.dump(histo, open(str(save_path), 'wb'))
    return


def midi_to_indexroll(midi_file, save_path):
    """Extract indexes of top note from midi.
       indexroll is np.array, shape is (song_len)
       (in the timing `t` that there is no notes, indexroll[t] = -1)
    """
    try:
        mid = pm.PrettyMIDI(str(midi_file))
        pianoroll = mid.get_piano_roll(fs=FS)
    except (KeyError, OSError, EOFError, ValueError,
            AttributeError, ZeroDivisionError) as e:
        print(str(midi_file), e)
        return

    index_roll = []
    for time_i in range(pianoroll.shape[1]):
        note_indexes = np.nonzero(pianoroll[:, time_i])[0]
        if len(note_indexes) == 0:
            # add blank note
            index_roll.append(-1)
        else:
            # add top note
            index_roll.append(note_indexes[-1])
    pickle.dump(np.array(index_roll), open(str(save_path), 'wb'))
    return


def lakh_tempo_change(original_root_dir, target_root_dir):
    process_lakh_dataset(original_root_dir=original_root_dir,
                         target_root_dir=target_root_dir,
                         original_suffix="mid",
                         target_suffix="mid",
                         process_file=change_tempo_midi)


def lakh_midi_to_histo(orignla_root_dir, target_root_dir):
    process_lakh_dataset(original_root_dir=original_root_dir,
                         target_root_dir=target_root_dir,
                         original_suffix='mid',
                         target_suffix='pickle',
                         process_file=midi_to_histo)


def lakh_midi_to_indexroll(orignla_root_dir, target_root_dir):
    process_lakh_dataset(original_root_dir=original_root_dir,
                         target_root_dir=target_root_dir,
                         original_suffix='mid',
                         target_suffix='pickle',
                         process_file=midi_to_indexroll)


def preprocess(original_root_dir, target_root_dir):
    tempo_root_dir = target_root_dir / "tempo_changed"
    histo_root_dir = target_root_dir / "histo"
    indexroll_root_dir = target_root_dir / "indexroll"

    # lakh_tempo_change(original_root_dir, tempo_root_dir)
    # lakh_midi_to_histo(tempo_root_dir, histo_root_dir)
    lakh_midi_to_indexroll(tempo_root_dir, indexroll_root_dir)


if __name__ == "__main__":
    original_root_dir = Path("/home/azuma/workspace/dataset/chord_lstm_data/debug")
    target_root_dir = Path("dataset")
    preprocess(original_root_dir, target_root_dir)
