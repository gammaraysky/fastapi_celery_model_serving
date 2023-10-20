"""
This module contains methods to chunk audio and RTTM data for training.

It processes audio files and their corresponding RTTM files to create
smaller audio chunks of equal duration and corresponding segmented RTTM
data.
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
    count_num_chunks,
    get_rttm_segments,
    get_signal,
    get_sr,
    segments_chunker,
    signal_chunker,
)

logger = logging.getLogger(__name__)


def train_chunking(
    wav_files: Dict[str, Callable],
    rttm_files: Dict[str, Callable],
    train_chunking_config: dict,
) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
    """Chunk audio and RTTM data for training.

    This function chunks audio and RTTM data for training purposes. It
    processes audio files and their corresponding RTTM files to create
    smaller audio chunks of equal duration and segmented RTTM data.

    The resulting chunked audio and RTTM data are returned as
    dictionaries where the keys represent the chunked file names
    (filename sans extension) and the values are callable functions for
    accessing the data.

    Args:
        wav_files (Dict[str, Callable]): A dictionary mapping file names
            to audio processing functions.
        rttm_files (Dict[str, Callable]): A dictionary mapping file
            names to RTTM processing functions.
        train_chunking_config (dict): A dictionary containing
            configuration parameters for chunking, including
            'sample_rate' and 'chunk_size_secs'.

    Returns:
        Tuple[Dict[str, Callable], Dict[str, Callable]]: A tuple
            containing two dictionaries:
            1. Dictionary mapping chunked audio file names to data
            transformation Callables, which will return the audio chunk
            content when evaluated.
            2. Dictionary mapping chunked RTTM file names to data
            transformation Callables, which will return the RTTM chunk
            content when evaluated.

    Raises:
        KeyError: If a matching RTTM file is missing for a given audio
        file.
    """
    # These are the file outputs, where the key will be the filename
    # sans extension, and the value will be the file data:
    # i.e for audio files, it is a tuple of the audio_signal (ndarray)
    # and the sample rate (int).
    # i.e for RTTM files, it is the text as string
    wav_outputs = {}
    rttm_outputs = {}

    # loop through each wave file in the dataset
    for wav_file_name, audio_file_callable in wav_files.items():
        # check for matching rttm file
        if wav_file_name not in list(rttm_files.keys()):
            logger.error("Missing RTTM for %s", wav_file_name)
            raise KeyError(
                "Missing RTTM file for given audio file: {}".format(wav_file_name)
            )
        else:
            rttm_file_callable = rttm_files[wav_file_name]

            # Calc how many chunks the audio file will produce
            # based on the chunk size configuration
            num_chunks = count_num_chunks(
                audio_file_callable,
                sample_rate=int(train_chunking_config["sample_rate"]),
                chunk_size=int(train_chunking_config["chunk_size_secs"]),
            )

            # Split audio file into chunks.
            # Produces the chunked signals as a list.
            # The chunked signals are returned as callable functions, so
            # that the data is not actually loaded at this point (Kedro
            # PartitionedDataset uses lazy loading by default) and we do
            # not run out of memory iterating through a large dataset.
            # By using callable functions, lazy saving is implemented
            # and each signal chunk is only processed and executed only
            # when it is being saved.
            signal_chunks = signal_chunker(
                audio_file_callable=audio_file_callable,
                file_id=wav_file_name,
                num_chunks=num_chunks,
                sample_rate=int(train_chunking_config["sample_rate"]),
                chunk_size=int(train_chunking_config["chunk_size_secs"]),
            )

            # logger.debug(signal_chunks)

            # signal chunks were a List[Tuple[file_id, signal_data]]
            # lets update wave output dict:
            for file_id_chunk, signal_chunk in signal_chunks:
                wav_outputs[file_id_chunk] = signal_chunk

            # Split RTTM annotations segments into chunks
            # Produces the chunked RTTM data as a list.
            # Also implements lazy saving as above.
            segments_chunks = segments_chunker(
                rttm_file_callable=rttm_file_callable,
                file_id=wav_file_name,
                num_chunks=num_chunks,
                chunk_size=int(train_chunking_config["chunk_size_secs"]),
            )

            # update rttm output dict
            for file_id_chunk, segments_chunk in segments_chunks:
                rttm_outputs[file_id_chunk] = segments_chunk

            logger.info("Processing: %s -> %s chunks", wav_file_name, num_chunks)

    return wav_outputs, rttm_outputs
