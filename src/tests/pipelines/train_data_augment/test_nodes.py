# Standard library imports
import os
import sys
import copy
import json
import logging
from typing import Callable, Dict
from unittest import mock

# Related third-party imports
import numpy as np
import pytest
import torch
from pydub import AudioSegment
import torch_audiomentations

# Local application/library specific imports
from klass.pipelines.train_data_augment.nodes import (
    train_data_augment,
    train_data_augment_validation,
    get_augmentation_from_config,
    augment_wavefile,
)

# Configure the logger to log messages with level INFO and above to stdout
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sample_train_wav_files = {
    "file1": lambda: (np.zeros(480000), 16000),
    "file2": lambda: (np.zeros(480000), 16000),
    "file3": lambda: (np.zeros(480000), 16000),
    "file4": lambda: (np.zeros(480000), 16000),
    "file5": lambda: (np.zeros(480000), 16000),
    "file6": lambda: (np.zeros(480000), 16000),
    "file7": lambda: (np.zeros(480000), 16000),
    "file8": lambda: (np.zeros(480000), 16000),
    "file9": lambda: (np.zeros(480000), 16000),
    "file10": lambda: (np.zeros(480000), 16000)
    # Add more sample data as needed
}

sample_train_rttm_files = {
    "file1": lambda: None,
    "file2": lambda: None,
    "file3": lambda: None,
    "file4": lambda: None,
    "file5": lambda: None,
    "file6": lambda: None,
    "file7": lambda: None,
    "file8": lambda: None,
    "file9": lambda: None,
    "file10": lambda: None
    # Add more sample data as needed
}

sample_chunking_sampling_config = {
    "sample_rate": 16000,
    "chunk_size_secs": 30,
    "mins_to_sample": 1500,
}

sample_augment_config = {
    "proportion_to_augment": 0.2,
    "augmentations": {
        "bgnoise": {
            "p": 1.0,
            "min_snr_in_db": 3.0,
            "max_snr_in_db": 30.0,
            "bg_paths": "/polyaxon-v1-data/workspaces/data/bg_noise",
        },
        "gain": {
            "p": 1.0,
            "min_gain_in_db": -15.0,
            "max_gain_in_db": 5.0,  # Add more augmentations as needed
        },
        "hipass": {"p": 1.0, "min_cutoff_freq": 20, "max_cutoff_freq": 500},
        "lopass": {"p": 1.0, "min_cutoff_freq": 3000, "max_cutoff_freq": 7500},
        "reverb": {
            "p": 0.0,
            "path": "/polyaxon-v1-data/workspaces/data/BUTreverb_rirs",
        },
    },
}


def test_train_data_augment():
    augmented_wav, augmented_rttm, augmented_json = train_data_augment(
        sample_train_wav_files,
        sample_train_rttm_files,
        sample_chunking_sampling_config,
        sample_augment_config,
    )

    # Check if the augmented dictionaries contain the same keys as the input dictionaries
    assert set(augmented_wav.keys()) == set(sample_train_wav_files.keys())
    assert set(augmented_rttm.keys()) == set(sample_train_rttm_files.keys())

    # Check if the proportion of files in augmented dictionaries is as expected
    proportion = sample_augment_config["proportion_to_augment"]
    num_files_to_augment = int(proportion * len(sample_train_wav_files))
    assert len(augmented_wav) == len(sample_train_wav_files)
    assert len(augmented_rttm) == len(sample_train_rttm_files)

    # Check if the augmented JSON report is a valid JSON string
    try:
        json.loads(augmented_json[0])
        assert True
    except json.JSONDecodeError:
        assert False


# Function to mock logging, since the tested function uses a logger
class MockLogger:
    def warning(self, *args, **kwargs):
        pass


logger = MockLogger()


@pytest.mark.parametrize("proportion_to_augment", [-0.1, 1.2])
def test_augmentation_config_validation_proportion_to_augment(proportion_to_augment):
    # Test 'proportion_to_augment' out of bounds
    # Update proportion_to_augment in the base config
    augment_config = copy.deepcopy(sample_augment_config)
    augment_config["proportion_to_augment"] = proportion_to_augment

    with pytest.raises(ValueError, match=r".*must be between 0.0 and 1.0"):
        train_data_augment_validation(
            sample_train_wav_files,
            sample_train_rttm_files,
            sample_chunking_sampling_config,
            augment_config,
        )


@pytest.mark.parametrize("p", [-0.1, 1.2])
def test_augmentation_config_validation_bgnoise_p(p):
    augment_config = copy.deepcopy(sample_augment_config)
    augment_config["augmentations"]["bgnoise"]["p"] = p

    with pytest.raises(ValueError, match=r".* must be between 0.0 and 1.0"):
        train_data_augment_validation(
            sample_train_wav_files,
            sample_train_rttm_files,
            sample_chunking_sampling_config,
            augment_config,
        )


@pytest.mark.parametrize("p", [-0.1, 1.2])
def test_augmentation_config_validation_gain_p(p):
    augment_config = copy.deepcopy(sample_augment_config)
    augment_config["augmentations"]["gain"]["p"] = p

    with pytest.raises(ValueError, match=r".* must be between 0.0 and 1.0"):
        train_data_augment_validation(
            sample_train_wav_files,
            sample_train_rttm_files,
            sample_chunking_sampling_config,
            augment_config,
        )


@pytest.mark.parametrize("p", [-0.1, 1.2])
def test_augmentation_config_validation_hipass_p(p):
    augment_config = copy.deepcopy(sample_augment_config)
    augment_config["augmentations"]["hipass"]["p"] = p

    with pytest.raises(ValueError, match=r".* must be between 0.0 and 1.0"):
        train_data_augment_validation(
            sample_train_wav_files,
            sample_train_rttm_files,
            sample_chunking_sampling_config,
            augment_config,
        )


@pytest.mark.parametrize("p", [-0.1, 1.2])
def test_augmentation_config_validation_lopass_p(p):
    augment_config = copy.deepcopy(sample_augment_config)
    augment_config["augmentations"]["lopass"]["p"] = p

    with pytest.raises(ValueError, match=r".* must be between 0.0 and 1.0"):
        train_data_augment_validation(
            sample_train_wav_files,
            sample_train_rttm_files,
            sample_chunking_sampling_config,
            augment_config,
        )


@pytest.mark.parametrize("p", [-0.1, 1.2])
def test_augmentation_config_validation_reverb_p(p):
    augment_config = copy.deepcopy(sample_augment_config)
    augment_config["augmentations"]["reverb"]["p"] = p

    with pytest.raises(ValueError, match=r".* must be between 0.0 and 1.0"):
        train_data_augment_validation(
            sample_train_wav_files,
            sample_train_rttm_files,
            sample_chunking_sampling_config,
            augment_config,
        )


# Test when the path exists


def test_bgnoise_path_exists(
    tmp_path,
):  # using tmp_path fixture provided by pytest for temporary directories
    dummy_path = tmp_path / "dummy_bgnoise"
    dummy_path.mkdir()  # create a temporary directory

    # construct the config
    augment_config = copy.deepcopy(sample_augment_config)
    augment_config["augmentations"]["bgnoise"]["bg_paths"] = dummy_path

    # this should not raise any error
    train_data_augment_validation(
        sample_train_wav_files,
        sample_train_rttm_files,
        sample_chunking_sampling_config,
        augment_config,
    )


# Test when the path does not exist


@pytest.mark.parametrize("bg_path", ["/path/that/does/not/exist"])
def test_bg_path_does_not_exist(bg_path):
    # construct the config
    augment_config = copy.deepcopy(sample_augment_config)
    augment_config["augmentations"]["bgnoise"]["bg_paths"] = bg_path

    with pytest.raises(
        ValueError,
        match=r'\["bgnoise"\]\["bg_paths"\] /path/that/does/not/exist does not exist',
    ):
        train_data_augment_validation(
            sample_train_wav_files,
            sample_train_rttm_files,
            sample_chunking_sampling_config,
            augment_config,
        )


def test_reverb_path_exists(
    tmp_path,
):  # using tmp_path fixture provided by pytest for temporary directories
    dummy_path = tmp_path / "dummy_bgnoise"
    dummy_path.mkdir()  # create a temporary directory

    # construct the config
    augment_config = copy.deepcopy(sample_augment_config)
    augment_config["augmentations"]["reverb"]["path"] = dummy_path

    # this should not raise any error
    train_data_augment_validation(
        sample_train_wav_files,
        sample_train_rttm_files,
        sample_chunking_sampling_config,
        augment_config,
    )


@pytest.mark.parametrize("path", ["/path/that/does/not/exist"])
def test_reverb_path_does_not_exist(path):
    # construct the config
    augment_config = copy.deepcopy(sample_augment_config)
    augment_config["augmentations"]["reverb"]["path"] = path

    with pytest.raises(
        ValueError,
        match=r'\["reverb"\]\["path"\] /path/that/does/not/exist does not exist',
    ):
        train_data_augment_validation(
            sample_train_wav_files,
            sample_train_rttm_files,
            sample_chunking_sampling_config,
            augment_config,
        )


# Mock classes for the augmentations
class AddBackgroundNoise:
    def __init__(self, background_paths, min_snr_in_db, max_snr_in_db, p):
        self.background_paths = background_paths
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.p = p


class Gain:
    def __init__(self, min_gain_in_db, max_gain_in_db, p):
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db
        self.p = p


class HighPassFilter:
    def __init__(self, min_cutoff_freq, max_cutoff_freq, p):
        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        self.p = p


class LowPassFilter:
    def __init__(self, min_cutoff_freq, max_cutoff_freq, p):
        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        self.p = p


class ApplyImpulseResponse:
    def __init__(self, ir_paths, p):
        self.ir_paths = ir_paths
        self.p = p


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms


def test_get_augmentation_from_config(tmp_path):
    dummy_path = tmp_path / "dummy_bgnoise"
    dummy_path.mkdir()  # create a temporary directory
    audio_file = dummy_path / "sample.wav"

    # Generate a 5-second silent audio clip using pydub
    silent_audio = AudioSegment.silent(duration=5000)  # duration in milliseconds

    # Export as WAV file
    silent_audio.export(audio_file, format="wav")

    # construct the config
    augment_config = copy.deepcopy(sample_augment_config["augmentations"])
    augment_config["bgnoise"]["bg_paths"] = dummy_path
    aug_callable = get_augmentation_from_config(augment_config)

    # Patch the Compose class from torch_audiomentations with your mock_Compose
    with mock.patch(
        "klass.pipelines.train_data_augment.nodes.Compose", new=Compose
    ), mock.patch(
        "klass.pipelines.train_data_augment.nodes.AddBackgroundNoise",
        new=AddBackgroundNoise,
    ), mock.patch(
        "klass.pipelines.train_data_augment.nodes.Gain", new=Gain
    ), mock.patch(
        "klass.pipelines.train_data_augment.nodes.HighPassFilter", new=HighPassFilter
    ), mock.patch(
        "klass.pipelines.train_data_augment.nodes.LowPassFilter", new=LowPassFilter
    ), mock.patch(
        "klass.pipelines.train_data_augment.nodes.ApplyImpulseResponse",
        new=ApplyImpulseResponse,
    ):
        aug_callable = get_augmentation_from_config(augment_config)
    assert isinstance(aug_callable, Compose)

    assert isinstance(aug_callable.transforms[0], AddBackgroundNoise)
    assert str(aug_callable.transforms[0].background_paths) == str(dummy_path)

    assert isinstance(aug_callable.transforms[1], Gain)
    assert aug_callable.transforms[1].min_gain_in_db == -15.0

    assert isinstance(aug_callable.transforms[2], HighPassFilter)
    assert aug_callable.transforms[2].min_cutoff_freq == 20

    assert isinstance(aug_callable.transforms[3], LowPassFilter)
    assert aug_callable.transforms[3].min_cutoff_freq == 3000

    assert isinstance(aug_callable.transforms[4], ApplyImpulseResponse)
    assert aug_callable.transforms[4].p == 0.0


def mock_train_wav_callable():
    return (torch.Tensor([0.5, 0.7, 0.9]), 44100)


def mock_get_augmentation_from_config(augmentations_to_apply):
    # Simulate an augmentation: just multiply the signal by 2 for this mock
    def apply_augmentation(signal, sample_rate):
        return signal * 2

    return apply_augmentation


class TestAugmentWavefile:
    @mock.patch("klass.pipelines.train_data_augment.nodes.logger")
    @mock.patch(
        "klass.pipelines.train_data_augment.nodes.get_augmentation_from_config",
        side_effect=mock_get_augmentation_from_config,
    )
    def test_successful_augmentation(self, mock_get_augmentation, mock_logger):
        aug_callable = augment_wavefile(
            mock_train_wav_callable, {"gain": {"min": -5, "max": 5}}
        )
        augmented_signal, sample_rate = aug_callable()

        # Validate the augmented signal
        # assert list(augmented_signal.numpy()) == [1.0, 1.4, 1.8]
        tolerance = 1e-5
        for actual, expected in zip(augmented_signal, [1.0, 1.4, 1.8]):
            assert abs(actual - expected) < tolerance
        assert sample_rate == 44100

        # Ensure logger did not register any errors
        mock_logger.error.assert_not_called()

    @mock.patch("klass.pipelines.train_data_augment.nodes.logger")
    @mock.patch(
        "klass.pipelines.train_data_augment.nodes.get_augmentation_from_config",
        side_effect=Exception("Some Error"),
    )
    def test_exception_during_augmentation(self, mock_get_augmentation, mock_logger):
        aug_callable = augment_wavefile(
            mock_train_wav_callable, {"gain": {"min": -5, "max": 5}}
        )
        augmented_signal, sample_rate = aug_callable()

        # Validate the original signal is returned
        tolerance = 1e-5
        for actual, expected in zip(augmented_signal, [0.5, 0.7, 0.9]):
            assert abs(actual - expected) < tolerance
        assert sample_rate == 44100

        # Ensure logger registered the error
        mock_logger.error.assert_called_once_with(
            "%s - Error during augmentation. Using original audio instead.",
            "Some Error",
        )
