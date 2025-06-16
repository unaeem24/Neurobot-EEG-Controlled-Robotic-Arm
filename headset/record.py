import datetime
import numpy as np
from mne import Info, create_info
from mne.io.array import RawArray
from pylsl import StreamInlet, resolve_streams, resolve_byprop

def get_info(SRATE=128) -> Info:
    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7',
                'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    info = create_info(
        sfreq=SRATE,
        ch_names=ch_names,
        ch_types=['eeg'] * len(ch_names)
    )
    return info

def record(SRATE=128, duration=9):
    # Resolve EEG stream and collect data (unchanged)
    print("looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG')
    inlet = StreamInlet(streams[0])

    buffer = []
    while True:
        if len(buffer) == SRATE * duration:  # SRATE * duration = 1152 samples (for 9 sec)
            break
        sample, _ = inlet.pull_sample()
        sample = [el / 1000000 for el in sample]  # Convert to microvolts
        buffer.append(sample)

    info = get_info(SRATE)
    raw = RawArray(np.array(buffer).T, info)  # Shape: (14, 1152)

    # Trim from 1152 → 1125 samples (center crop)
    data = raw.get_data()  # Shape: (14, 1152)
    start_idx = (data.shape[1] - 1125) // 2  # Centered crop
    trimmed_data = data[:, start_idx : start_idx + 1125]  # Shape: (14, 1125)

    # Verify shape
    assert trimmed_data.shape == (14, 1125), f"Expected (14, 1125), got {trimmed_data.shape}"

    return trimmed_data  # Return (14, 1125) for ATC-Net

if __name__ == "__main__":
    record()
