import multiprocessing as mp
import os

import noisereduce
import numpy as np
import numpy.typing as npt
from pydub import AudioSegment
from python_speech_features import delta, mfcc
from scipy.io import wavfile

from src import config


def process(do_parallel=True):
    # load files

    filenames = os.listdir(config.PATH_DATA_RAW)
    filepaths = [
        os.path.join(config.PATH_DATA_RAW, filename) for filename in filenames
    ]

    if do_parallel:
        pool = mp.Pool(mp.cpu_count())
        features = [
            pool.apply(_parallel_worker, args=(filepath,))
            for filepath in filepaths
        ]
    else:
        features = [_parallel_worker(filepath) for filepath in filepaths]
    print("Finished")


def _parallel_worker(filepath) -> tuple[str, npt.NDArray]:
    """contains parallelisable operations to import and preprocess single audio files
    returns tuple (filename, feature array) or (filename, None) if feature extraction failed
    """
    filename_with_ext = os.path.split(filepath)[-1]
    filename, _ = os.path.splitext(filename_with_ext)
    print(f"Processing {filename}...")

    # convert to wav
    if not _wav_exists(filename_with_ext):
        filepath = _convert_to_wav(filepath)

    # load wave file
    rate, data = wavfile.read(filepath)

    # if stereo, crop second channel
    if len(data.shape) == 2 and data.shape[1] == 2:
        data = data[:, 0]

    # perform noise reduction
    data = noisereduce.reduce_noise(y=data, sr=rate)

    # perform voice activity detection and remove nonvoice segments
    data = _remove_nonvoice_segments(data)

    if len(data) == 0:
        return (filename, None)

    # Store
    filename_with_ext = os.path.split(filepath)[
        -1
    ]  # update this after converted to .wav
    filepath_preprocessed = os.path.join(
        config.PATH_DATA_PREPROCESSED, filename_with_ext
    )
    wavfile.write(filepath_preprocessed, data=data, rate=rate)

    # Extract features
    mfcc_features = mfcc(
        data,
        rate,
        winlen=0.020,
        preemph=0.95,
        numcep=20,
        nfft=1024,
        ceplifter=15,
        highfreq=6000,
        nfilt=55,
        appendEnergy=False,
    )

    # result is a list of shape (len/winlen, numcep), so for each window of size winlen, we have numcep channels

    # Now calculate delta features
    delta_features = delta(mfcc_features, 2)  # calculating delta
    combined_features = np.hstack((mfcc_features, delta_features))
    return (filename, combined_features)


def _wav_exists(
    filename_ext, converted_directory: str = config.PATH_DATA_CONVERTED
) -> bool:
    return os.path.exists(os.path.join(converted_directory, filename_ext))


def _convert_to_wav(
    filepath: str, converted_directory: str = config.PATH_DATA_CONVERTED
) -> str:
    """Imports the file at the given path, converts it to wav format and stores it to 'converted' directory.
    If already in wav format, the file is only copied to the 'converted' directory.

    Args:
        filepath (str): path to audio file (with a common file ending like opus, ogg, wav)
        converted_directory (str): path to directory where converted files are stored

    Returns: filepath of converted file
    """
    filename_with_ext = os.path.split(filepath)[-1]
    filename, _ = os.path.splitext(filename_with_ext)
    filename_converted = filename + ".wav"
    filepath_converted = os.path.join(converted_directory, filename_converted)

    audio = AudioSegment.from_file(filepath)
    audio.export(filepath_converted, format="wav")
    return filepath_converted


def _remove_nonvoice_segments(arr, min_length=1000, min_value_rel=0.01):
    """Naive implementation: identify all segments with absolute activation lower than a threshold and a given min length."""
    # TODO We have to normalise the amplitude?
    # TODO Try webrtcvad

    arr = np.asarray(arr)  # enforce numpy array

    arr_norm = arr / arr.max()
    arr_binary = arr_norm >= min_value_rel

    idx_pairwise_unequal = arr_binary[1:] != arr_binary[:-1]
    idx_end = np.append(np.where(idx_pairwise_unequal), len(arr_binary) - 1)
    seg_length = np.diff(np.append(-1, idx_end))  # lengths of segments

    # filter for value zero
    mask_valuefilter = arr_binary[idx_end] == 0
    idx_end_valuefilter = idx_end[mask_valuefilter]
    seg_length_valuefilter = seg_length[mask_valuefilter]

    # filter for min_length
    mask_lenfilter = seg_length_valuefilter >= min_length
    idx_end_valuefilter_lenfilter = idx_end_valuefilter[mask_lenfilter]
    seg_length_valuefilter_lenfilter = seg_length_valuefilter[mask_lenfilter]

    mask = np.repeat(True, len(arr_binary))
    for end, length in zip(
        idx_end_valuefilter_lenfilter, seg_length_valuefilter_lenfilter
    ):
        mask[end - length + 1 : end + 1] = 0

    return arr[mask]


if __name__ == "__main__":
    process(do_parallel=True)
