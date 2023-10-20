"""Unit Tests for train_data_chunking pipeline
Usage:
cd vad/
pytest klass/src/tests/pipelines/train_data_chunking/test_nodes.py
"""
import os
import sys

sys.path.append("/polyaxon-v1-data/workspaces/hanafi/154/vad/klass/src/")

from typing import Callable, Dict
from unittest import mock

# from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from klass.pipelines.train_data_chunking.nodes import train_chunking


def test_train_chunking_normal():
    """Runs unit test: 430s input wave file (synthetic), with some
    speech segments that span across chunks, when chunked with chunk
    size of 100s.

    Cases covered:
    - does chunker treat remainder/excess properly
    - does chunker miss any speech segments at the chunk boundaries
    - if a segment is very long and spans across multiple chunks, is
      this handled properly
    """

    def get_audiofile():
        return (np.zeros(16000 * 430), 16000)

    def get_rttm():
        return """SPEAKER file1 1 0.1 10.0 <NA> <NA> SPEECH <NA> <NA>
SPEAKER file1 1 80.0 40.0 <NA> <NA> SPEECH <NA> <NA>
SPEAKER file1 1 150.0 200.0 <NA> <NA> SPEECH <NA> <NA>
"""

    wav_files = {"file1": get_audiofile}
    rttm_files = {"file1": get_rttm}
    train_chunking_config = {"sample_rate": 16000, "chunk_size_secs": 100}

    wav_outputs, rttm_outputs = train_chunking(
        wav_files, rttm_files, train_chunking_config
    )

    expected_filenames = [
        "file1_0-100",
        "file1_100-200",
        "file1_200-300",
        "file1_300-400",
    ]

    expected_outputs = [
        """SPEAKER file1_0-100 1 0.1 10.0 <NA> <NA> SPEECH <NA> <NA>
SPEAKER file1_0-100 1 80.0 20.0 <NA> <NA> SPEECH <NA> <NA>""",
        """SPEAKER file1_100-200 1 0 20.0 <NA> <NA> SPEECH <NA> <NA>
SPEAKER file1_100-200 1 50.0 50.0 <NA> <NA> SPEECH <NA> <NA>""",
        """SPEAKER file1_200-300 1 0 100 <NA> <NA> SPEECH <NA> <NA>""",
        """SPEAKER file1_300-400 1 0 50.0 <NA> <NA> SPEECH <NA> <NA>""",
    ]

    assert list(wav_outputs.keys()) == expected_filenames
    assert list(rttm_outputs.keys()) == expected_filenames

    for output, expected in zip(rttm_outputs.values(), expected_outputs):
        assert output().strip() == expected.strip()


def test_train_chunking_missing_rttm():
    """Runs unit test:

    Cases covered:
    - get warning if missing rttm file
    """

    def get_audiofile():
        return (np.zeros(16000 * 430), 16000)

    def get_rttm():
        return """SPEAKER file1 1 0.1 10.0 <NA> <NA> SPEECH <NA> <NA>
SPEAKER file1 1 80.0 40.0 <NA> <NA> SPEECH <NA> <NA>
SPEAKER file1 1 150.0 200.0 <NA> <NA> SPEECH <NA> <NA>
"""

    wav_files = {"file1": get_audiofile, "file2": get_audiofile}
    rttm_files = {"file1": get_rttm}
    train_chunking_config = {"sample_rate": 16000, "chunk_size_secs": 100}

    with pytest.raises(KeyError) as exc_info:
        wav_outputs, rttm_outputs = train_chunking(
            wav_files, rttm_files, train_chunking_config
        )

    assert str(exc_info.value) == "'Missing RTTM file for given audio file: file2'"


def test_train_chunking_shortwavefile():
    """Runs unit test: 30s input wave file (synthetic), when config
    chunk size set to 100s.

    Cases covered:
    - if original file too short
    """

    def get_audiofile():
        return (np.zeros(16000 * 30), 16000)

    def get_rttm():
        return """SPEAKER file1 1 0.1 10.0 <NA> <NA> SPEECH <NA> <NA>
SPEAKER file1 1 20.0 30.0 <NA> <NA> SPEECH <NA> <NA>
"""

    wav_files = {"file1": get_audiofile}
    rttm_files = {"file1": get_rttm}

    train_chunking_config = {"sample_rate": 16000, "chunk_size_secs": 100}

    wav_outputs, rttm_outputs = train_chunking(
        wav_files, rttm_files, train_chunking_config
    )

    assert list(wav_outputs.keys()) == []
    assert list(rttm_outputs.keys()) == []
