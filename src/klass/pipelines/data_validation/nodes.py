"""
This is a boilerplate pipeline 'data_validation'
generated using Kedro 0.18.12
"""
import json
import logging
from typing import Callable, Dict, List, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def validate_and_convert_wav_files(
    source_wav_files: Dict[str, Callable],
    soundfile_info: Dict[str, Callable],
    rttm_files: Dict[str, Callable],
) -> Tuple[Dict[str, Callable], Dict[str, Callable], List[str]]:
    """
    Validate, Convert, and Check Annotations for WAV Files.

    This function validates provided WAV files against specific
    criteria, checks their corresponding annotations, and converts them
    if necessary to meet required audio specifications. It returns a
    dictionary of callables for the WAV files that passed the validation
    (and were possibly converted), and a report in JSON format outlining
    files that do not meet each criterion. If a WAV file has annotation
    errors, it will not be included in the returned dictionary.

    Audio Transformation:
    - WAV files not in mono will be transformed to mono.
    - WAV files not at 16 kHz will be resampled to 16 kHz.

    Annotation Checks (No modifications, only reported):
    - Files must contain speech segments. Files without speech will not
      be included.
    - Annotations must not exceed WAV file duration by more than 1
      second. These files will not be included.

    Args:
        source_wav_files (Dict[str, Callable]): A dictionary mapping WAV
            filenames to callables that provide WAV file data.
        soundfile_info (Dict[str, Callable]): A dictionary mapping
            WAV filenames to callables that provide soundfile
            information.
        rttm_files (Dict[str, Callable]): A dictionary mapping WAV
            filenames to callables that provide the RTTM content as a
            string.

    Returns:
        Tuple[Dict[str, Callable], Dict[str, Callable], List[str]]:
            A tuple containing:
            - A dictionary mapping WAV filenames to callables that
              provide the RTTM content as a string.
            - A dictionary containing validated and optionally converted
              WAV files. WAV files with annotation errors are excluded.
            - A list with a single JSON-formatted string reporting files
              not meeting the specified criteria.

    Example of returned JSON:
    {
        "non_mono": ['ES2010d.Array1-01', 'ES2010d.Array1-02', ...],
        "non_16khz": [...],
        "files without speech": [...],
        "annotations_exceeding_duration": [...],
    }
    """
    if not check_matching_keys(source_wav_files, rttm_files):
        logger.warning("Keys in source WAV files and RTTM do not match.")

    non_mono_files = []
    non_16khz_files = []
    files_without_speech = []
    annotations_exceeding_duration = []

    destination_wav_files = {}
    destination_rttm_files = {}

    # Loop through wav files
    for wav_file_name, soundfile_info_callable in soundfile_info.items():
        # Get soundfile info
        soundfile_info = soundfile_info_callable()

        target_sample_rate = 16000
        wav_file_callable = source_wav_files[wav_file_name]

        wav_file_data = None

        # If missing key:
        if rttm_files.get(wav_file_name) is None:
            continue

        # If no speech
        if has_no_speech_segments(rttm_files[wav_file_name]):
            files_without_speech.append(wav_file_name)
            continue

        # If annotations too long
        if annotations_too_long(soundfile_info_callable, rttm_files[wav_file_name]):
            annotations_exceeding_duration.append(wav_file_name)
            continue

        # If not mono
        if soundfile_info.channels > 1:
            non_mono_files.append(wav_file_name)
            wav_file_data = convert_wav_file_to_mono_channel(wav_file_callable())

        # If not 16khz
        if soundfile_info.samplerate != target_sample_rate:
            non_16khz_files.append(wav_file_name)
            wav_file_data = convert_wav_file_to_16khz(wav_file_callable())

        # If the WAV file data was changed due to conversion, wrap it
        # into a callable, otherwise, use the original callable
        destination_wav_files[wav_file_name] = (
            lambda wav_data=wav_file_data: wav_data
            if wav_file_data is not None
            else wav_file_callable()
        )

        destination_rttm_files[wav_file_name] = rttm_files[wav_file_name]

        non_mono_files = sorted(non_mono_files)
        non_16khz_files = sorted(non_16khz_files)
        files_without_speech = sorted(files_without_speech)
        annotations_exceeding_duration = sorted(annotations_exceeding_duration)

        format_check_report = {
            "non_mono": non_mono_files,
            "non_16khz": non_16khz_files,
            "files_without_speech": files_without_speech,
            "annotations_exceeding_duration": annotations_exceeding_duration,
        }

        json_contents = json.dumps(format_check_report)

    return destination_rttm_files, destination_wav_files, [json_contents]


def check_matching_keys(
    source_wav_files: Dict[str, Callable], rttm_files: Dict[str, Callable]
) -> bool:
    """
    Checks if the keys of two dictionaries match.

    Args:
        source_wav_files (Dict[str, Callable]): A dictionary mapping WAV
            filenames to callables providing WAV file data.
        rttm_files (Dict[str, Callable]): A dictionary mapping WAV
            filenames to callables providing RTTM content.

    Returns:
        bool: True if keys of both dictionaries match, otherwise logs
            an error message and returns False.
    """
    source_keys = set(source_wav_files.keys())
    rttm_keys = set(rttm_files.keys())

    if source_keys == rttm_keys:
        return True
    else:
        missing_in_source = rttm_keys - source_keys
        missing_in_rttm = source_keys - rttm_keys

        warning_message = []
        if missing_in_source:
            warning_message.append(
                f"Keys present in RTTM but missing in source WAV files: {', '.join(missing_in_source)}."
            )
        if missing_in_rttm:
            warning_message.append(
                f"Keys present in source WAV files but missing in RTTM: {', '.join(missing_in_rttm)}."
            )

        logger.warning(" ".join(warning_message))
        return False


def convert_wav_file_to_mono_channel(
    wav_file: Tuple[np.ndarray, int]
) -> Tuple[np.ndarray, int]:
    """
    Convert a given WAV file to mono by selecting the leftmost channel.

    Args:
        wav_file (Tuple[np.ndarray, int]): A tuple containing the WAV signal (as a numpy array)
                                          and its sample rate.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the converted mono WAV signal
                               and its original sample rate.

    Raises:
        IOError: If there's an error processing the WAV file.
    """
    try:
        signal, samplerate = wav_file
        # Check if the signal is mono
        if len(signal.shape) == 1:
            return (signal, samplerate)
        leftmost_signal = signal[:, 0]
        return (leftmost_signal, samplerate)
    except Exception as e:
        raise IOError(f"Error occurred while processing the WAV file: {e}")


def convert_wav_file_to_16khz(
    wav_file: Tuple[np.ndarray, int]
) -> Tuple[np.ndarray, int]:
    """
    Convert a given WAV file to a sample rate of 16 kHz.

    Args:
        wav_file (Tuple[np.ndarray, int]): A tuple containing the WAV signal (as a numpy array)
                                          and its original sample rate.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the resampled WAV signal at 16 kHz
                               and the new sample rate (16000).

    Raises:
        IOError: If there's an error converting the WAV file to 16 kHz.
    """
    try:
        signal, samplerate = wav_file
        target_sr = 16000
        signal_resamp = librosa.resample(
            y=signal, orig_sr=samplerate, target_sr=target_sr
        )
        return (signal_resamp, target_sr)

    except Exception as e:
        raise IOError(f"Error occurred while converting the WAV file to 16 kHz: {e}")


def has_no_speech_segments(rttm_file_callable):
    """
    Checks if an RTTM file (provided as a callable returning lines) has no speech segments at all.

    Args:
        rttm_file_callable (Callable): A callable that provides lines from the RTTM file.

    Returns:
        bool: True if there are no speech segments in the file, False otherwise.
    """
    segments_text = rttm_file_callable()
    segments = segments_text.split("\n")

    for segment in segments:
        if "SPEAKER" in segment:
            return False  # at least one speech segment found

    # for line in rttm_file_callable():
    #     parts = line.strip().split()
    #     if parts and parts[0] == "SPEAKER":
    #         return False  # At least one speech segment found

    return True  # No speech segments found


def get_end_time_of_last_segment(rttm_file_callable: callable) -> float:
    """
    Get the end time of the last speech segment from a callable RTTM
    content.

    Args:
        rttm_file_callable (callable): A callable that provides the RTTM
        content as a string.

    Returns:
        float: End time of the last speech segment.
    """
    # Split the RTTM content into lines
    lines = rttm_file_callable().split("\n")

    # Filter out empty lines (if any)
    lines = [line for line in lines if line.strip()]

    # Get the last line from the list
    last_line = lines[-1]

    # Split the line into parts
    parts = last_line.split()

    # The fourth column (index 3) is the start time
    start_time = float(parts[3])

    # The fifth column (index 4) is the duration
    duration = float(parts[4])

    # Calculate end time
    end_time = start_time + duration

    return end_time


def annotations_too_long(
    soundfile_info_callable: callable, rttm_file_callable: callable
) -> bool:
    """
    Check if annotations in the RTTM file exceed the WAV file's duration
    by more than 1 second.

    Args:
        soundfile_info_callable (callable): A callable that provides the
        soundfile info.
        rttm_file_callable (callable): A callable that provides the RTTM
        content as a string.

    Returns:
        bool: True if annotations are too long, False otherwise.
    """
    soundfile_info = soundfile_info_callable()

    # Directly use the duration attribute from soundfile_info
    wav_duration = soundfile_info.duration

    last_annotation_end_time = get_end_time_of_last_segment(rttm_file_callable)

    # Check if the difference between the last annotation's end time and the WAV's duration exceeds 1 second
    return (last_annotation_end_time - wav_duration) > 1.0
