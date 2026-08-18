# Channels of interest
INCLUDED_CHANNELS = [
    'EEG FP1',
    'EEG FP2',
    'EEG F3',
    'EEG F4',
    'EEG C3',
    'EEG C4',
    'EEG P3',
    'EEG P4',
    'EEG O1',
    'EEG O2',
    'EEG F7',
    'EEG F8',
    'EEG T3',
    'EEG T4',
    'EEG T5',
    'EEG T6',
    'EEG FZ',
    'EEG CZ',
    'EEG PZ']

CHBMIT_INCLUDED_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

# Resampling frequency
FREQUENCY = 200

# All seizure labels available in TUH
ALL_LABEL_DICT = {'fnsz': 0, 'gnsz': 1, 'spsz': 2, 'cpsz': 3,
                  'absz': 4, 'tnsz': 5, 'tcsz': 6, 'mysz': 7}
