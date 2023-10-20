import json
from typing import Dict, List, Callable
import logging

import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import Mock

from klass.pipelines.data_validation.nodes import (
    validate_and_convert_wav_files,
    convert_wav_file_to_mono_channel,
    convert_wav_file_to_16khz,
    has_no_speech_segments,
    get_end_time_of_last_segment,
    annotations_too_long,
    check_matching_keys,
)


def test_convert_to_mono_successful():
    # Create a fake stereo signal
    stereo_signal = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    samplerate = 44100

    converted_signal, converted_samplerate = convert_wav_file_to_mono_channel(
        (stereo_signal, samplerate)
    )

    expected_signal = np.array([1, 3, 5, 7])
    assert np.array_equal(converted_signal, expected_signal)
    assert converted_samplerate == samplerate


def test_convert_mono_signal_unchanged():
    # Create a fake mono signal
    mono_signal = np.array([1, 3, 5, 7])
    samplerate = 44100

    converted_signal, converted_samplerate = convert_wav_file_to_mono_channel(
        (mono_signal, samplerate)
    )

    assert np.array_equal(converted_signal, mono_signal)
    assert converted_samplerate == samplerate


def test_convert_mono_signal_error_handling():
    with pytest.raises(IOError, match="Error occurred while processing the WAV file"):
        convert_wav_file_to_mono_channel(([], 44100))


def test_resample_to_16khz():
    # Create a fake signal
    sr = 44100
    t = np.linspace(0, 1, sr)
    signal = np.sin(2 * np.pi * 220 * t)  # A 220Hz sine wave

    converted_signal, converted_samplerate = convert_wav_file_to_16khz((signal, sr))

    # Check that the output sample rate is 16kHz
    assert converted_samplerate == 16000

    # Check the length of the resampled signal is correct
    expected_length = len(signal) * 16000 // sr
    assert len(converted_signal) == expected_length


def test_already_16khz():
    # Create a fake signal
    sr = 16000
    t = np.linspace(0, 1, sr)
    signal = np.sin(2 * np.pi * 220 * t)  # A 220Hz sine wave

    converted_signal, converted_samplerate = convert_wav_file_to_16khz((signal, sr))

    # Check that the output sample rate is 16kHz
    assert converted_samplerate == sr

    # Ensure that the signal was not changed
    assert np.array_equal(converted_signal, signal)


def test_resampling_error_handling():
    with pytest.raises(
        IOError, match="Error occurred while converting the WAV file to 16 kHz"
    ):
        convert_wav_file_to_16khz(([], 44100))


def mock_rttm_file(lines: List[str]) -> callable:
    """Utility function to mock an RTTM file callable"""
    return lambda: "\n".join(lines)


# Test cases using parametrize
@pytest.mark.parametrize(
    "rttm_lines, expected",
    [
        # Case with speech segment
        (
            [
                "SPEAKER file_name 1 2.3 1.2 <NA> <NA> speaker_id <NA> <NA>",
                "SOME_OTHER_INFO ...",
            ],
            False,
        ),
        # Empty case
        ([], True),
    ],
)
def test_has_no_speech_segments(rttm_lines, expected):
    result = has_no_speech_segments(mock_rttm_file(rttm_lines))
    assert result == expected


def mock_rttm_content(lines: str) -> callable:
    """Utility function to mock an RTTM content callable"""
    return lambda: lines


@pytest.mark.parametrize(
    "rttm_content, expected_end_time",
    [
        # Single entry
        ("SPEAKER file_name 1 2.3 1.2 <NA> <NA> speaker_id <NA> <NA>\n", 3.5),
        # Multiple entries
        (
            "SPEAKER file_name 1 2.3 1.2 <NA> <NA> speaker_id <NA> <NA>\n"
            "SPEAKER file_name 1 5.0 1.5 <NA> <NA> speaker_id <NA> <NA>\n",
            6.5,
        ),
        # Entry with trailing newline
        ("SPEAKER file_name 1 2.3 1.2 <NA> <NA> speaker_id <NA> <NA>\n\n", 3.5),
        # Complex duration
        (
            "SPEAKER file_name 1 293.8399999999999 6.160000000000082 <NA> <NA> speaker_id <NA> <NA>\n",
            300.0,
        ),
    ],
)
def test_get_end_time_of_last_segment(rttm_content, expected_end_time):
    result = get_end_time_of_last_segment(mock_rttm_content(rttm_content))
    assert result == expected_end_time


def mock_soundfile_info(duration: float) -> callable:
    """Utility function to mock a soundfile info callable with given duration."""
    return lambda: type("SoundfileInfo", (object,), {"duration": duration})()


@pytest.mark.parametrize(
    "wav_duration, rttm_content, expected",
    [
        # Annotation ends exactly at wav file's duration
        (3.5, "SPEAKER file_name 1 2.3 1.2 <NA> <NA> speaker_id <NA> <NA>\n", False),
        # Annotation ends more than 1 second after wav file's duration
        (3.5, "SPEAKER file_name 1 2.3 2.3 <NA> <NA> speaker_id <NA> <NA>\n", True),
        # Annotation ends 0.5 seconds after wav file's duration
        (3.5, "SPEAKER file_name 1 2.3 1.7 <NA> <NA> speaker_id <NA> <NA>\n", False),
    ],
)
def test_annotations_too_long(wav_duration, rttm_content, expected):
    result = annotations_too_long(
        mock_soundfile_info(wav_duration), mock_rttm_content(rttm_content)
    )
    assert result == expected


def test_matching_keys():
    source_wav_files = {
        "file1": lambda: None,
        "file2": lambda: None,
        "file3": lambda: None,
    }
    rttm_files = {"file1": lambda: None, "file2": lambda: None, "file3": lambda: None}

    result = check_matching_keys(source_wav_files, rttm_files)
    assert result is True


def test_mismatch_keys_missing_in_source(mocker):
    source_wav_files = {"file1": lambda: None, "file3": lambda: None}
    rttm_files = {"file1": lambda: None, "file2": lambda: None, "file3": lambda: None}

    mock_warning = mocker.patch("klass.pipelines.data_validation.nodes.logger.warning")
    check_matching_keys(source_wav_files, rttm_files)

    mock_warning.assert_called_with(
        "Keys present in RTTM but missing in source WAV files: file2."
    )


def test_mismatch_keys_missing_in_rttm(mocker):
    source_wav_files = {
        "file1": lambda: None,
        "file2": lambda: None,
        "file3": lambda: None,
    }
    rttm_files = {"file1": lambda: None, "file3": lambda: None}

    mock_warning = mocker.patch("klass.pipelines.data_validation.nodes.logger.warning")
    check_matching_keys(source_wav_files, rttm_files)

    mock_warning.assert_called_with(
        "Keys present in source WAV files but missing in RTTM: file2."
    )


def test_mismatch_both_sides(mocker):
    source_wav_files = {"file1": lambda: None, "file2": lambda: None}
    rttm_files = {"file1": lambda: None, "file3": lambda: None}

    mock_warning = mocker.patch("klass.pipelines.data_validation.nodes.logger.warning")
    check_matching_keys(source_wav_files, rttm_files)

    expected_message = (
        "Keys present in RTTM but missing in source WAV files: file3. "
        "Keys present in source WAV files but missing in RTTM: file2."
    )
    mock_warning.assert_called_with(expected_message)


def test_empty_dictionaries():
    source_wav_files = {}
    rttm_files = {}

    result = check_matching_keys(source_wav_files, rttm_files)
    assert result is True
