"""snr_distributions.py

Checks across train/val/test sets of a dataset.
Workflow involves generating SNR values per audio file, using its RTTM
speech segments as signal and non-speech as noise, a csv of SNR values
across the dataset/split is saved, and summary stats and similarity
tests can be run across dataset splits to check for consistency/drift.
"""
# import sys
# import os
import logging

# from pathlib import Path
from typing import Dict, List, Union  # Tuple, Optional

# import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.vad.data_prep import speech_segments as ss

logger = logging.getLogger(__name__)


def get_snrs_across_data_split(
    dataset_split: Dict, return_df: bool = False
) -> Union[List, pd.DataFrame]:
    """Generate SNRs across entire dataset/split.
    e.g. Ali_far train subset.

    Args:
        dataset_split (Dict): Dict with the following structure:
            {
                'file1' : {
                    'audio_path' : path to audio file,
                    'rttm_path' : path to audio file,
                    'segments' : List of segments start and end time
                        tuples e.g. [(1.0, 2.1), (5.0, 9.7), ...]
                }
                'file2' : { ... }
                'file3' : { ... }
                ...
            }
        return_df (bool): If true, returns DataFrame. If false, returns
            list. Defaults to False.

    Returns:
        Union[List, pd.DataFrame]: Returns either SNRs as a List or
            Dataframe.

    """

    data_samples = list(dataset_split.keys())

    snrs = []
    for data_sample in tqdm(data_samples):
        snrs.append(get_snr(dataset_split[data_sample]))

    if return_df:
        return pd.DataFrame(
            {
                "fileid": data_samples,
                "snr": snrs,
            }
        )
    else:
        return snrs


def get_snr(data_sample: Dict) -> float:
    """get_snr _summary_

    Args:
        data_sample (Dict): Dict containing keys 'audio_path' and
            'segments', that respectively has path to audio file and a
            list of tuples of start and end times for each speech
            segment

    Returns:
        float: SNR in dB.
    """
    duration_secs = sf.info(data_sample["audio_path"]).duration
    speech_segments = data_sample["segments"]

    nonspeech_segments = ss.invert_segments(speech_segments, duration_secs)

    audio_data, sample_rate = sf.read(data_sample["audio_path"])

    speech_signal = ss.concat_signal_segments(
        speech_segments, (audio_data, sample_rate)
    )
    noise_signal = ss.concat_signal_segments(
        nonspeech_segments, (audio_data, sample_rate)
    )

    snr = calc_snr(speech_signal, noise_signal)

    return snr


def signal_power(x):
    return np.average(np.array(x) ** 2)


def calc_snr(signal, noise):
    pow_signal = signal_power(signal)
    pow_noise = signal_power(noise)
    return 10 * np.log10((pow_signal / pow_noise))
