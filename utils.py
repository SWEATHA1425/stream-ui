import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Pad or truncate to 100 time steps
    if mfcc.shape[1] < 100:
        pad_width = 100 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :100]

    # Reshape to match CNN input: (1, 40, 100, 1)
    mfcc = mfcc.reshape(1, 40, 100, 1)
    return mfcc
