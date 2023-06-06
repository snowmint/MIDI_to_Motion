import sys
import scipy
import pretty_midi
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


def read_midi(f_path, resample_fs):
    """
    Load MIDI file
    """
    midi_data = pretty_midi.PrettyMIDI(f_path)

    # Get the piano roll representation of the MIDI file
    piano_roll = midi_data.get_piano_roll(fs=resample_fs)
    piano_roll[piano_roll > 0] = 1
    piano_roll = np.transpose(piano_roll)
    
    # piano_roll.shape = (T, 128)
    return piano_roll


def read_csv(f_path, resample_fs):
    """
    Load CSV file
    """
    n_frame, _, _, sf, _ = pd.read_csv(f_path, nrows=5, names=['attr', 'value'], on_bad_lines='skip')['value']
    n_frame, sf = int(n_frame), int(sf)
    csv_motion = pd.read_csv(f_path, skiprows=5, on_bad_lines='skip')
    csv_motion = csv_motion.iloc[: n_frame-1, :]
    # csv_motion.shape = (T, 102)
    
    col_id_LANK_X = list(csv_motion.columns).index("LANK_X")
    
    # Interpolate '0.0'
    csv_motion = csv_motion.replace(0.0, np.NaN)
    csv_motion = csv_motion.interpolate(axis=0)
    csv_motion = csv_motion.bfill()
    csv_motion = csv_motion.ffill()
    
    # Resample
    resample_len = n_frame // sf * resample_fs
    csv_motion = scipy.signal.resample(csv_motion, resample_len)
    
    # Center with respect to "LANK"
    center = [csv_motion[:, col_id_LANK_X].copy(),
              csv_motion[:, col_id_LANK_X+1].copy(),
              csv_motion[:, col_id_LANK_X+2].copy()]
    for col in range(csv_motion.shape[1]):
        csv_motion[:, col] -= center[col % 3]
    csv_motion /= np.max(np.abs(csv_motion))
    
    # csv_motion.shape = (T, 102)
    return csv_motion


if __name__ == "__main__":
    import torch
    midi_FS = 20
    motion_FS = 20
    
    performer = "vio01"
    piece = "Bach1_S1_T1"
    
    # Test midi data
    midi_f_path = f"./UoE_violin_midi/violin_midi/{performer}/{performer}_{piece}.mid"    
    midi_data = read_midi(midi_f_path, midi_FS)
    pd.DataFrame(midi_data).to_csv(f"midi_{midi_FS}FS.csv")
    print("midi.shape:", midi_data.shape)
    
    # Test csv
    csv_f_path = f"./UoE_violin_midi/performance_motion/{performer}/{performer}_{piece}.csv"
    motion_data = read_csv(csv_f_path, motion_FS)
    # pd.DataFrame(motion_data).to_csv(f"motion_{motion_FS}FS.csv")
    print("motion.shape:", motion_data.shape)
    

