"""
This is a boilerplate pipeline 'train_data_sampling'
generated using Kedro 0.18.12
"""

import hashlib
import logging
import math
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple, Union

import librosa
import numpy as np
import soundfile as sf

from klass.extras.datasets.rttm import parse_rttm_line
from klass.extras.datasets.train_data_chunking_sampling import (
    train_data_sampling_gethash,
    fileid_hasher,
)


logger = logging.getLogger(__name__)


def train_data_sampling(
    train_wav_files: Dict[str, Callable],
    train_rttm_files: Dict[str, Callable],
    sampling_config: dict,
) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
    """Sample a specified amount of total minutes from a dataset of
    chunked wave and annotation RTTM files, using hashed filenames for
    sorting order and selection.

    Args:
        train_wav_files (Dict[str, Callable]): A dictionary mapping file
            IDs to wav file processing functions.
        train_rttm_files (Dict[str, Callable]): A dictionary mapping
            file IDs to RTTM file processing functions.
        sampling_config (dict): A dictionary containing configuration
            parameters for sampling, including 'chunk_size_secs' and
            'mins_to_sample'.

    Returns:
        Tuple[Dict[str, Callable], Dict[str, Callable]]: A tuple
            containing two dictionaries:
            1. Dictionary mapping selected wave file IDs to processing
            functions.
            2. Dictionary mapping selected RTTM file IDs to processing
            functions.
    """
    chunk_size_secs = int(sampling_config["chunk_size_secs"])
    mins_to_sample = int(sampling_config["mins_to_sample"])

    ### RUN CHECKS TO ENSURE SAMPLING WILL WORK ###
    train_wav_files, train_rttm_files = train_data_sampling_validation(
        train_wav_files, train_rttm_files, sampling_config
    )

    ### RANDOM SAMPLING ###
    # hash filenames in the dataset
    # (to use as a 'shuffle' but ensure reproducibility across systems)
    file_ids = list(train_wav_files.keys())
    hashed_fileids = train_data_sampling_gethash(file_ids)
    shuffled = sorted(hashed_fileids.keys())

    # is mins_to_sample divisible by chunk_size_secs, round off if need, sample until mins reached>exceeds
    num_files_needed = math.ceil(mins_to_sample / (chunk_size_secs / 60))

    # select necessary files
    selected = shuffled[:num_files_needed]
    wav_outputs, rttm_outputs = {}, {}
    for hashstring in selected:
        which_fileid = hashed_fileids[hashstring]
        wav_outputs[which_fileid] = train_wav_files[which_fileid]
        rttm_outputs[which_fileid] = train_rttm_files[which_fileid]

    return wav_outputs, rttm_outputs


def train_data_sampling_validation(
    train_wav_files: Dict[str, Callable],
    train_rttm_files: Dict[str, Callable],
    sampling_config: dict,
):
    """Perform validation checks prior to sampling audio data.

    This function ensures the consistency and compatibility of audio and
    annotation RTTM pairs for sampling. It also verifies that the audio
    files meet the specified chunk size and sample rate as defined in
    the sampling configuration. Additionally, it checks if the dataset
    has sufficient audio duration to fulfill the sampling request.

    Args:
        train_wav_files (Dict[str, Callable]): A dictionary mapping file
            IDs to audio processing functions.
        train_rttm_files (Dict[str, Callable]): A dictionary mapping
            file IDs to RTTM processing functions.
        sampling_config (dict): A dictionary containing configuration
            parameters for sampling, including 'sample_rate',
            'chunk_size_secs', and 'mins_to_sample'.

    Returns:
        Tuple[Dict[str, Callable], Dict[str, Callable]]: A tuple
            containing two dictionaries:
            1. Dictionary mapping valid audio file IDs to processing
            functions.
            2. Dictionary mapping valid RTTM file IDs to processing
            functions.

    Raises:
        ValueError: If any of the validation checks fail, including
            missing files, incorrect path, or incorrect configuration
            parameters.
    """
    sample_rate = int(sampling_config["sample_rate"])
    chunk_size_secs = int(sampling_config["chunk_size_secs"])
    mins_to_sample = int(sampling_config["mins_to_sample"])
    # if mins_to_sample <= 0:
    #    raise ValueError("mins_to_sample must be > 0")
    if chunk_size_secs <= 0:
        raise ValueError("chunk_size_secs must be > 0")

    # check audio/rttm pair matches, drops mismatched files
    common_fnames = set(train_wav_files.keys()) & set(train_rttm_files.keys())
    common_fnames = list(common_fnames)

    missing_rttms = set(train_wav_files.keys()) - set(train_rttm_files.keys())
    missing_wavs = set(train_rttm_files.keys()) - set(train_wav_files.keys())

    if len(missing_rttms) > 0:
        logger.warning("Audio files with missing RTTM: %s", missing_rttms)
    if len(missing_wavs) > 0:
        logger.warning("RTTM files with missing audio: %s", missing_wavs)

    # check every file matches chunk_size_secs, sample_rate
    # can skip this if we want faster processing
    train_wav_files_valid = {}
    train_rttm_files_valid = {}
    for fname, audio_callable in train_wav_files.items():
        if fname not in common_fnames:
            logger.warning("Missing RTTM file for audio file %s. Skipping", fname)
            continue
        signal, sr = audio_callable()
        duration = int(len(signal) / sr)
        if sr != sample_rate:
            logger.warning(
                "Audio file sample rate %s does not match train_sampling_config['sample_rate'] %s. Skipping %s",
                sr,
                sample_rate,
                fname,
            )
            continue
        if duration != chunk_size_secs:
            logger.warning(
                "Audio file duration %s s does not match train_sampling_config['chunk_size_secs'] %s s. Skipping %s",
                duration,
                chunk_size_secs,
                fname,
            )
            continue
        del signal, sr
        train_wav_files_valid[fname] = audio_callable
        train_rttm_files_valid[fname] = train_rttm_files[fname]

    if len(list(train_wav_files_valid.keys())) == 0:
        raise ValueError(
            "Check dataset or config. Either missing files, incorrect path, or incorrect train_data_sampler config params specified."
        )

    # check sufficient duration in dataset to meet request
    valid_fnames = list(train_rttm_files_valid.keys())
    dataset_available_duration_mins = len(valid_fnames) * chunk_size_secs / 60
    if dataset_available_duration_mins < mins_to_sample:
        logger.error(
            "Available minutes: %s, Requested minutes: %s",
            dataset_available_duration_mins,
            mins_to_sample,
        )
        raise ValueError("Dataset has insufficient duration to fulfil sample request.")

    return train_wav_files_valid, train_rttm_files_valid
