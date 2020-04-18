import pickle
from collections import Counter
from pathlib import Path

import mido
import numpy as np
import pretty_midi as pm
from tqdm import tqdm


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


def tempo_change(original_root_dir, target_root_dir):
    process_lakh_dataset(original_root_dir=original_root_dir,
                         target_root_dir=target_root_dir,
                         original_suffix="mid",
                         target_suffix="mid",
                         process_file=change_tempo_midi)


def preprocess(original_root_dir, target_root_dir):
    tempo_change(original_root_dir, target_root_dir / "tempo_changed")


if __name__ == "__main__":
    original_root_dir = Path("/home/azuma/workspace/dataset/chord_lstm_data/lmd_matched/A")
    target_root_dir = Path("dataset")
    preprocess(original_root_dir, target_root_dir)
