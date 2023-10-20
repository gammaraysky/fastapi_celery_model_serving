""" Shared helper modules for train_data_chunking and
train_data_sampling pipelines
"""
import hashlib
import logging
from typing import Callable, Dict, List, Tuple, Union

import numpy as np

from klass.extras.datasets.rttm import parse_rttm_line

logger = logging.getLogger(__name__)


def get_signal(audio_file_callable: Callable):
    """
    Returns a callable function to extract the audio signal from an audio file.

    Args:
        audio_file_callable (Callable): A callable function that returns a tuple
            containing the audio signal as the first element and the sample rate as
            the second element.

    Returns:
        Callable: A callable function that extracts and returns the audio
            signal when invoked.
    """

    def signal_callable():
        return audio_file_callable()[0]

    return signal_callable


def get_sr(audio_file_callable: Callable):
    """
    Returns a callable function to extract the sample rate from an audio file.

    Args:
        audio_file_callable (Callable): A callable function that returns a tuple
            containing the audio signal as the first element and the sample rate as
            the second element.

    Returns:
        Callable: A callable function that extracts and returns the sample
            rate when invoked.

    """

    def sr_callable():
        return audio_file_callable()[1]

    return sr_callable


def get_rttm_segments(rttm_file_callable: Callable):
    """
    Returns a callable function to parse and extract segments from an RTTM file.

    Args:
        rttm_file_callable (Callable): A callable function that returns the content
            of an RTTM file as a string.

    Returns:
        Callable: A callable function that parses and extracts segments
            from the RTTM file content when invoked.

    """

    def segment_callable():
        return [
            parse_rttm_line(line)
            for line in rttm_file_callable().split("\n")
            if len(line) > 0
        ]

    return segment_callable


def count_num_chunks(
    audio_file_callable: Callable, sample_rate: int = 16000, chunk_size: int = 300
) -> int:
    """Given an audio signal array and its sample rate, calculates how
    many chunks will be produced if we chunk into clips of length
    chunk_size in seconds."""
    signal, _ = audio_file_callable()

    if len(signal) <= 0:
        return 0
    num_chunks = len(signal) // (chunk_size * sample_rate)
    del signal, _

    return num_chunks


def get_chunk_signal(
    audio_file_callable: Callable,
    file_id: str,
    chunk_start: int,
    chunk_end: int,
    sample_rate: int = 16000,
) -> Callable:
    """Given an audio file and its config, returns a Tuple of the
    output file_id and signal chunk for that specified chunk, as a
    callable"""
    key_label = f"{file_id}_{chunk_start}-{chunk_end}"

    def get_chunk_signal_callable() -> Tuple[str, np.ndarray]:
        signal, sr = audio_file_callable()

        return (signal[chunk_start * sample_rate : chunk_end * sample_rate], sr)

    return (key_label, get_chunk_signal_callable)


def signal_chunker(
    audio_file_callable: Callable,
    file_id: str,
    num_chunks: int,
    sample_rate: int = 16000,
    chunk_size: int = 300,
) -> Dict[str, np.ndarray]:
    """
    Function returns a dictionary of signal chunks in blocks of
    chunk_size (seconds).
    Default chunk_size = 300 seconds and sample_rate = 16kHz.
    Any remaider signal not within a chunk_size will be discarded
    Args:
        signal (np.ndarray) : samples from audio file in
            array form
        file_id (str) : file id (name without extension) of wave file
        num_chunks (int) : number of chunks to produce
        sample_rate (int) : sample rate of audio
        chunk_size (int) : audio chunk size required in seconds

    Returns: Tuple of:
        signal_chunks_dict (dict) : Dictionary containing audio file
            name as keys and signal array of chunk size as values
        num_chunks (int) : Number of chunks the audio was split into.
    """
    signal_chunks = []

    for i in range(num_chunks):
        # naming the dict keys based on the input file and the chunk no e.g.
        # 'R0001_M0001_SPK0094' + '_300-600' for the 300 to 600s chunk.
        chunk_start = i * chunk_size
        chunk_end = (i + 1) * chunk_size

        signal_chunks.append(
            get_chunk_signal(
                audio_file_callable, file_id, chunk_start, chunk_end, sample_rate
            )
        )

    return signal_chunks


def get_chunk_segments(
    rttm_file_callable: Callable, file_id: str, chunk_start: int, chunk_end: int
):
    """FIXME explain what the RTTM fields consists of

    Args:
        rttm_file_callable (Callable): _description_
        file_id (str): _description_
        chunk_start (int): _description_
        chunk_end (int): _description_

    Returns:
        _type_: _description_
    """
    outfileid = f"{file_id}_{chunk_start}-{chunk_end}"  # "{}-{}".format(i * chunk_size, (i + 1) * chunk_size)

    def get_chunk_segments_callable():
        segments_text = rttm_file_callable()
        segments_text = segments_text.split("\n")
        segments = [parse_rttm_line(line) for line in segments_text if len(line) > 0]

        # in this chunk, gather all the segments that are within the chunk_start and _end time, e.g. 300-600s, and offset the values
        segments_this_chunk = []
        for onset, duration, _, speaker_id in segments:
            # CASES TO ACCEPT:
            # 1) if this segment starts before current chunk, but
            #    overlaps into current chunk, correct the values
            if onset <= chunk_start and onset + duration > chunk_start:
                # e.g. onset = 270, chunk_start = 300
                # duration = 50
                diff = chunk_start - onset
                onset = chunk_start
                duration = duration - diff

            # 2) if this segment ends after current chunk, but overlaps
            #    with current chunk, correct the values
            if onset < chunk_end and onset + duration >= chunk_end:
                diff2 = chunk_end - onset
                duration = diff2

            # 3) if this segment is fully within current chunk
            if onset >= chunk_start and onset + duration <= chunk_end:
                segments_this_chunk.append(
                    (
                        (onset - chunk_start),
                        duration,
                        outfileid,
                        speaker_id,
                    )
                )

        # reformat from List[Tuple] into str:
        segments_this_chunk_text = ""
        for onset, duration, file_id_chunk, speaker_id in segments_this_chunk:
            # segments_this_chunk_text += f"{onset}, {duration}, {file_id_chunk}, {speaker_id}\n"

            fields = [
                "SPEAKER",
                file_id_chunk,
                "1",
                str(round(onset, 2)),
                str(round(duration, 2)),
                "<NA>",
                "<NA>",
                speaker_id,
                "<NA>",
                "<NA>",
            ]
            line = " ".join(fields)
            segments_this_chunk_text += f"{line}\n"

        # logger.info(segments_this_chunk_text)
        return segments_this_chunk_text

    return (outfileid, get_chunk_segments_callable)


def segments_chunker(
    rttm_file_callable: Callable,
    file_id: str,
    num_chunks: int,
    chunk_size: int = 300,
    # input_rttm_path: str, out_rttm_path: str, start_time: int, end_time: int
):
    """
    Generates a .rttm file by extracting a portion of a full .rttm
    file given a start and end time interval
    Args:
        input_rttm_path (str) : input path of rttm file
        out_rttm_path (str) : output path of rttm file
        start_time (int) : Portion of rttm file to start extraction from
        end_time (int) : Portion of rttm file to extract  until

    """
    # logger.info("CHUNKER...\n")
    segments_chunks = []

    # iterate through num_chunks
    for i in range(num_chunks):
        chunk_start = i * chunk_size
        chunk_end = (i + 1) * chunk_size

        segments_chunks.append(
            get_chunk_segments(rttm_file_callable, file_id, chunk_start, chunk_end)
        )

    return segments_chunks


def train_data_sampling_gethash(file_ids: List[str]) -> dict:
    """train_data_sampling_gethash
    Given a list of filenames, generate hash of the filenames.

    Args:
        file_ids (List[str]): List of file IDs of the audio dataset

    Returns:
        dict: A lookup dict with hash alphanumeric string as keys, and original
            file_ids as values.
    """
    hashed_fileids = {}
    for file_id in file_ids:
        hashed_fileids[fileid_hasher(file_id)] = file_id

    return hashed_fileids


def fileid_hasher(file_id: str):
    """
    Function converts a filename given in string format into a
        Secure Hash Algorithm 256-bit hexadecimal string
    Args:
        file_id (str) : Audio file name in string
    Returns:
        sha256_hash.hexdigest() (str) : string object in hexadecimal
            digits
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(file_id.encode("utf-8"))
    logger.debug("%s successfully hashed", file_id)

    return sha256_hash.hexdigest()
