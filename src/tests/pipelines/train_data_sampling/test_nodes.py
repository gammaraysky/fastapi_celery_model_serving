"""Unit Tests for train_data_sampling pipeline
Usage:
cd vad/
pytest klass/src/tests/pipelines/train_data_sampling/test_nodes.py
"""
import os
import sys

sys.path.append("/polyaxon-v1-data/workspaces/hanafi/154/vad/klass/src/")

import logging
from typing import Callable, Dict
from unittest import mock

import numpy as np
import pytest

from klass.pipelines.train_data_sampling.nodes import train_data_sampling

# from unittest.mock import MagicMock, Mock
import logging

# Configure the logger to log messages with level INFO and above to stdout
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@pytest.fixture
def wav_files():
    def get_30s_audiofile():
        return (np.zeros(16000 * 30), 16000)

    wav_files = {
        "001": get_30s_audiofile,
        "002": get_30s_audiofile,
        "003": get_30s_audiofile,
        "004": get_30s_audiofile,
        "005": get_30s_audiofile,
        "006": get_30s_audiofile,
        "007": get_30s_audiofile,
        "008": get_30s_audiofile,
        "009": get_30s_audiofile,
        "010": get_30s_audiofile,
    }
    return wav_files


@pytest.fixture
def rttm_files():
    def get_30s_rttm():
        return """SPEAKER file1 1 10.0 10.0 <NA> <NA> SPEECH <NA> <NA>"""

    rttm_files = {
        "001": get_30s_rttm,
        "002": get_30s_rttm,
        "003": get_30s_rttm,
        "004": get_30s_rttm,
        "005": get_30s_rttm,
        "006": get_30s_rttm,
        "007": get_30s_rttm,
        "008": get_30s_rttm,
        "009": get_30s_rttm,
        "010": get_30s_rttm,
    }
    return rttm_files


def test_train_sampling_normal(wav_files, rttm_files):
    """Tests train_data_sampling node.

    Cases:
    - Given input of 10 wave files of 30s, returns a sampled set
    fulfilling 2mins.

    - Given chunk_size_secs <= 0, it should error out

    - Given mins_to_sample <= 0, it should error out

    """

    train_sampling_config = {
        "sample_rate": 16000,
        "chunk_size_secs": 30,
        "mins_to_sample": 2,
    }

    sampled_wav_outputs, sampled_rttm_outputs = train_data_sampling(
        wav_files, rttm_files, train_sampling_config
    )

    logger.debug(sampled_wav_outputs.keys())
    logger.debug(sampled_rttm_outputs.keys())

    expected_filenames = ["010", "009", "005", "007"]

    assert list(sampled_wav_outputs.keys()) == expected_filenames
    assert list(sampled_rttm_outputs.keys()) == expected_filenames


def test_train_sampling_mismatch_chunk_size_secs(wav_files, rttm_files):
    """Tests train_data_sampling node.

    Cases:
    - Given chunk_size_secs <= 0, it should error out
    """
    train_sampling_config = {
        "sample_rate": 16000,
        "chunk_size_secs": 2,
        "mins_to_sample": 10,
    }

    with pytest.raises(ValueError) as exc_info:
        sampled_wav_outputs, sampled_rttm_outputs = train_data_sampling(
            wav_files, rttm_files, train_sampling_config
        )
    logging.info(str(exc_info.value))
    assert (
        str(exc_info.value)
        == "Check dataset or config. Either missing files, incorrect path, or incorrect train_data_sampler config params specified."
    )


def test_train_sampling_bad_chunk_size_zero(wav_files, rttm_files):
    """Tests train_data_sampling node.

    Cases:
    - Given chunk_size_secs <= 0, it should error out
    """
    train_sampling_config = {
        "sample_rate": 16000,
        "chunk_size_secs": 0,
        "mins_to_sample": 10,
    }

    with pytest.raises(ValueError) as exc_info:
        sampled_wav_outputs, sampled_rttm_outputs = train_data_sampling(
            wav_files, rttm_files, train_sampling_config
        )
    logging.info(str(exc_info.value))
    assert str(exc_info.value) == "chunk_size_secs must be > 0"


def test_train_sampling_bad_mins_to_sample(wav_files, rttm_files):
    """Tests train_data_sampling node.

    Cases:
    - Given chunk_size_secs <= 0, it should error out
    """
    train_sampling_config = {
        "sample_rate": 16000,
        "chunk_size_secs": 10,
        "mins_to_sample": 0,
    }

    with pytest.raises(ValueError) as exc_info:
        sampled_wav_outputs, sampled_rttm_outputs = train_data_sampling(
            wav_files, rttm_files, train_sampling_config
        )

    assert str(exc_info.value) == "mins_to_sample must be > 0"


def test_train_sampling_incorrect_chunk_size_secs(wav_files, rttm_files):
    """Tests train_data_sampling node.

    Cases:
    - Given chunk_size_secs <= 0, it should error out
    """
    train_sampling_config = {
        "sample_rate": 16000,
        "chunk_size_secs": 10,
        "mins_to_sample": 2,
    }

    with pytest.raises(ValueError) as exc_info:
        sampled_wav_outputs, sampled_rttm_outputs = train_data_sampling(
            wav_files, rttm_files, train_sampling_config
        )

    assert (
        str(exc_info.value)
        == "Check dataset or config. Either missing files, incorrect path, or incorrect train_data_sampler config params specified."
    )


def test_train_sampling_incorrect_sample_rate(wav_files, rttm_files):
    """Tests train_data_sampling node.

    Cases:
    - Given chunk_size_secs <= 0, it should error out
    """
    train_sampling_config = {
        "sample_rate": 8000,
        "chunk_size_secs": 30,
        "mins_to_sample": 2,
    }

    with pytest.raises(ValueError) as exc_info:
        sampled_wav_outputs, sampled_rttm_outputs = train_data_sampling(
            wav_files, rttm_files, train_sampling_config
        )

    assert (
        str(exc_info.value)
        == "Check dataset or config. Either missing files, incorrect path, or incorrect train_data_sampler config params specified."
    )


def test_train_sampling_insufficient_mins(wav_files, rttm_files):
    """Tests train_data_sampling node.

    Cases:
    - Given chunk_size_secs <= 0, it should error out
    """
    train_sampling_config = {
        "sample_rate": 16000,
        "chunk_size_secs": 30,
        "mins_to_sample": 200,
    }

    with pytest.raises(ValueError) as exc_info:
        sampled_wav_outputs, sampled_rttm_outputs = train_data_sampling(
            wav_files, rttm_files, train_sampling_config
        )

    assert (
        str(exc_info.value)
        == "Dataset has insufficient duration to fulfil sample request."
    )


def test_train_sampling_missing_rttm(wav_files, rttm_files):
    """Tests train_data_sampling node.

    Cases:
    - Missing RTTMs for some files
    """
    train_sampling_config = {
        "sample_rate": 16000,
        "chunk_size_secs": 30,
        "mins_to_sample": 2,
    }

    [rttm_files.pop(fileid) for fileid in ["010", "009", "005"]]

    sampled_wav_outputs, sampled_rttm_outputs = train_data_sampling(
        wav_files, rttm_files, train_sampling_config
    )

    assert list(sampled_wav_outputs.keys()) == ["007", "001", "003", "004"]


def test_train_sampling_missing_wav(wav_files, rttm_files):
    """Tests train_data_sampling node.

    Cases:
    - Missing WAVs for some RTTM files
    """
    train_sampling_config = {
        "sample_rate": 16000,
        "chunk_size_secs": 30,
        "mins_to_sample": 2,
    }

    [wav_files.pop(fileid) for fileid in ["010", "007", "005"]]

    sampled_wav_outputs, sampled_rttm_outputs = train_data_sampling(
        wav_files, rttm_files, train_sampling_config
    )

    assert list(sampled_wav_outputs.keys()) == ["009", "001", "003", "004"]


# def test_train_sampling_shortwavefile():
#     """Runs unit test: 30s input wave file (synthetic), when config
#     chunk size set to 100s.

#     Cases covered:
#     - if original file too short
#     """

#     def get_audiofile():
#         return (np.zeros(16000 * 30), 16000)

#     def get_rttm():
#         return """SPEAKER file1 1 0.1 10.0 <NA> <NA> SPEECH <NA> <NA>
# SPEAKER file1 1 20.0 30.0 <NA> <NA> SPEECH <NA> <NA>
# """

#     wav_files = {"file1": get_audiofile}
#     rttm_files = {"file1": get_rttm}

#     train_sampling_config = {"sample_rate": 16000, "chunk_size_secs": 100}

#     sampled_wav_outputs, sampled_rttm_outputs = train_sampling(
#         wav_files, rttm_files, train_sampling_config
#     )

#     assert list(sampled_wav_outputs.keys()) == []
#     assert list(sampled_rttm_outputs.keys()) == []


# test_train_sampling_shortwavefile()

# test_train_sampling_normal()
