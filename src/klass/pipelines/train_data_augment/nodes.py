"""
This is a boilerplate pipeline 'train_data_augment'
generated using Kedro 0.18.12
"""
import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import torch
from torch_audiomentations import (
    AddBackgroundNoise,
    ApplyImpulseResponse,
    Compose,
    Gain,
    HighPassFilter,
    LowPassFilter,
)

from klass.extras.datasets.train_data_chunking_sampling import (
    train_data_sampling_gethash,
)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


logger = logging.getLogger(__name__)


def train_data_augment(
    train_wav_files: Dict[str, Callable],
    train_rttm_files: Dict[str, Callable],
    chunking_sampling_config: dict,
    augment_config: dict,
) -> Tuple[Dict[str, Callable], Dict[str, Callable], List[Dict[str, Any]]]:
    """Augment a portion of the train dataset while preserving the rest.

    Args:
        train_wav_files (Dict[str, Callable]): A dictionary mapping file
            names to corresponding WAV file loaders. The train dataset
            should have been chunked and sampled beforehand.
        train_rttm_files (Dict[str, Callable]): A dictionary mapping
            file names to corresponding RTTM file loaders.
        chunking_sampling_config (dict): Configuration for chunking and
            sampling.
        augment_config (dict): Configuration for data augmentation.

    Returns:
        Tuple[
            Dict[str, Callable],
            Dict[str, Callable],
            List[Dict[str, Any]]
        ] : A tuple containing:
            - A dictionary mapping the augmented train WAV files to the
              augmented wave file Callables
            - A dictionary mapping the augmented train RTTM files to the
              augmented RTTM file Callables
            - A JSON report file detailing which files were augmented

    Example:
        Given a train dataset with 30s clips ('file0001' to 'file1000'),
        this function returns the entire dataset with a specified
        proportion (e.g., 0.2) of files augmented randomly, using hashed
        filenames for sorting and selection order.
        It also generates a JSON report in the processed folder,
        detailing the filenames of the augmented files.

    Note:
        Audio files for background noise need to be longer than chunk
        size specified.
    """
    logger.info(augment_config)

    proportion = float(augment_config["proportion_to_augment"])
    augmentations_to_apply = augment_config["augmentations"]

    # logger.info("proportion: %s", proportion)

    # RUN CHECKS TO ENSURE SAMPLING WILL WORK ###
    train_wav_files, train_rttm_files = train_data_augment_validation(
        train_wav_files=train_wav_files,
        train_rttm_files=train_rttm_files,
        chunking_sampling_config=chunking_sampling_config,
        augment_config=augment_config,
    )

    ### RANDOM SAMPLING ###
    # hash filenames in the dataset
    # (to use as a 'shuffle' but ensure reproducibility across systems)
    file_ids = list(train_wav_files.keys())
    hashed_fileids = train_data_sampling_gethash(file_ids)
    shuffled = sorted(hashed_fileids.keys())

    num_files_needed = int(proportion * len(file_ids))

    # select necessary files
    selected = shuffled[:num_files_needed]
    originals = shuffled[num_files_needed:]
    # logger.info("SELECTED: %s", selected)
    # logger.info("ORIGINALS: %s", originals)

    wav_outputs, rttm_outputs = {}, {}
    augmented_filenames = []
    for hashstring in selected:
        which_fileid = hashed_fileids[hashstring]
        logger.info("AUGMENTING: %s", which_fileid)
        # augment here
        wav_outputs[which_fileid] = augment_wavefile(
            train_wav_files[which_fileid], augmentations_to_apply
        )
        augmented_filenames.append(which_fileid)
        rttm_outputs[which_fileid] = train_rttm_files[which_fileid]

    # augmented_filenames = "\n".join(augmented_filenames)
    # logger.info(augmented_filenames)
    augmented_filenames = {"augmented": augmented_filenames}
    augmented_json = json.dumps(augmented_filenames)

    for hashstring in originals:
        which_fileid = hashed_fileids[hashstring]
        wav_outputs[which_fileid] = train_wav_files[which_fileid]
        rttm_outputs[which_fileid] = train_rttm_files[which_fileid]

    return wav_outputs, rttm_outputs, [augmented_json]


def train_data_augment_validation(
    train_wav_files: Dict[str, Callable],
    train_rttm_files: Dict[str, Callable],
    chunking_sampling_config: dict,
    augment_config: dict,
):
    """Perform validation checks before data augmentation.

    This function ensures the integrity of the audio/RTTM pairs,
    validating that they match. It also verifies that the input WAV
    files have the correct chunk size and sample rate, as defined in the
    augment configuration YAML.

    Args:
        train_wav_files (Dict[str, Callable]): A dictionary mapping file
            names to corresponding WAV file loaders.
        train_rttm_files (Dict[str, Callable]): A dictionary mapping
            file names to corresponding RTTM file loaders.
        chunking_sampling_config (dict): Configuration for chunking and
            sampling.
        augment_config (dict): Configuration for data augmentation.

    Returns:
        Tuple[dict, float]: A tuple containing:
            - A dictionary representing the validated train WAV dataset
            - A dictionary representing the validated train RTTM dataset
    """
    try:
        float(augment_config["proportion_to_augment"])
        float(augment_config["augmentations"]["gain"]["p"])
        float(augment_config["augmentations"]["hipass"]["p"])
        float(augment_config["augmentations"]["lopass"]["p"])
        float(augment_config["augmentations"]["reverb"]["p"])
        float(augment_config["augmentations"]["bgnoise"]["min_snr_in_db"])
        float(augment_config["augmentations"]["bgnoise"]["max_snr_in_db"])
        float(augment_config["augmentations"]["gain"]["min_gain_in_db"])
        float(augment_config["augmentations"]["gain"]["max_gain_in_db"])
        int(augment_config["augmentations"]["hipass"]["min_cutoff_freq"])
        int(augment_config["augmentations"]["hipass"]["max_cutoff_freq"])
        int(augment_config["augmentations"]["lopass"]["min_cutoff_freq"])
        int(augment_config["augmentations"]["lopass"]["max_cutoff_freq"])

    except Exception as exc_info:
        raise TypeError(exc_info) from exc_info

    if (
        float(augment_config["proportion_to_augment"]) < 0
        or float(augment_config["proportion_to_augment"]) > 1
    ):
        raise ValueError('["proportion_to_augment"] must be between 0.0 and 1.0')

    if (
        float(augment_config["augmentations"]["bgnoise"]["p"]) < 0
        or float(augment_config["augmentations"]["bgnoise"]["p"]) > 1
    ):
        raise ValueError('["bgnoise"]["p"] must be between 0.0 and 1.0')

    if (
        float(augment_config["augmentations"]["gain"]["p"]) < 0
        or float(augment_config["augmentations"]["gain"]["p"]) > 1
    ):
        raise ValueError('["gain"]["p"] must be between 0.0 and 1.0')

    if (
        float(augment_config["augmentations"]["hipass"]["p"]) < 0
        or float(augment_config["augmentations"]["hipass"]["p"]) > 1
    ):
        raise ValueError('["hipass"]["p"] must be between 0.0 and 1.0')

    if (
        float(augment_config["augmentations"]["lopass"]["p"]) < 0
        or float(augment_config["augmentations"]["lopass"]["p"]) > 1
    ):
        raise ValueError('["lopass"]["p"] must be between 0.0 and 1.0')

    if (
        float(augment_config["augmentations"]["reverb"]["p"]) < 0
        or float(augment_config["augmentations"]["reverb"]["p"]) > 1
    ):
        raise ValueError('["reverb"]["p"] must be between 0.0 and 1.0')

    if Path(augment_config["augmentations"]["bgnoise"]["bg_paths"]).exists() is False:
        raise ValueError(
            f'["bgnoise"]["bg_paths"] {augment_config["augmentations"]["bgnoise"]["bg_paths"]} does not exist'
        )

    if Path(augment_config["augmentations"]["reverb"]["path"]).exists() is False:
        raise ValueError(
            f'["reverb"]["path"] {augment_config["augmentations"]["reverb"]["path"]} does not exist'
        )

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
        if sr != chunking_sampling_config["sample_rate"]:
            logger.warning(
                "Audio file sample rate %s does not match train_augment_config['sample_rate'] %s. Skipping",
                sr,
                chunking_sampling_config["sample_rate"],
            )
            continue
        if duration != chunking_sampling_config["chunk_size_secs"]:
            logger.warning(
                "Audio file duration %s s does not match train_augment_config['chunk_size_secs'] %s s. Skipping",
                duration,
                chunking_sampling_config["chunk_size_secs"],
            )
            continue
        del signal, sr
        train_wav_files_valid[fname] = audio_callable
        train_rttm_files_valid[fname] = train_rttm_files[fname]

    if len(list(train_wav_files_valid.keys())) == 0:
        raise ValueError(
            "Check dataset or config. Either missing files, incorrect path, "
            "or incorrect train_data_sampler config params specified."
        )

    return train_wav_files_valid, train_rttm_files_valid


def augment_wavefile(
    train_wav_callable: Callable,
    augmentations_to_apply: dict,
) -> Callable:
    """Create an augmentation callable for a WAV file.

    This function generates an augmentation callable based on the
    provided augmentation configuration. The callable takes no arguments
    and returns a tuple containing the augmented waveform array and the
    sample rate.

    If an error occurs during augmentation, the original audio data is
    returned, and the error is logged.

    Args:
        train_wav_callable (Callable): A callable that loads a WAV file
            and returns the waveform array and sample rate.
        augmentations_to_apply (Dict[str, dict]): A dictionary
            specifying the augmentations to be applied, where keys are
            augmentation names and values are their corresponding
            configuration dictionaries.

    Returns:
        Callable: An augmentation callable that applies the specified
            augmentations to the WAV file loaded by `train_wav_callable`
            and returns the augmented waveform and sample rate.

    """

    def augment_callable():
        signal, sample_rate = train_wav_callable()
        try:
            apply_augmentation = get_augmentation_from_config(augmentations_to_apply)

            signal = np.reshape(signal, (1, 1, -1))

            augmented_signal_array = torch.reshape(
                apply_augmentation(torch.Tensor(signal), sample_rate=sample_rate), (-1,)
            ).numpy()

            # logger.info(augmented_signal_array.shape)

            return (augmented_signal_array, sample_rate)

        except Exception as exc:
            logger.error(
                "%s - Error during augmentation. Using original audio instead.",
                str(exc),
            )
            return (train_wav_callable()[0], sample_rate)

    return augment_callable


def get_augmentation_from_config(aug_cfg: dict) -> Callable:
    """Generate an augmentation callable from the provided
    configuration.

    This function creates an augmentation callable based on the
    configuration provided in the `aug_cfg` dictionary. The callable
    applies a series of audio augmentations, such as adding background
    noise, adjusting gain, applying high-pass and low-pass filters, and
    applying impulse responses.

    Args:
        aug_cfg (dict): A dictionary containing configuration parameters
            for audio augmentations.

    Returns:
        Callable: An augmentation callable that applies the specified
            audio augmentations based on the provided configuration.

    Example:
        To apply a set of audio augmentations defined in a configuration
        dictionary, call this function with the configuration as an
        argument. The resulting callable can be used to augment audio
        data according to the specified configuration.
    """
    augmentations_callable = Compose(
        transforms=[
            AddBackgroundNoise(
                background_paths=aug_cfg["bgnoise"]["bg_paths"],
                min_snr_in_db=aug_cfg["bgnoise"]["min_snr_in_db"],
                max_snr_in_db=aug_cfg["bgnoise"]["max_snr_in_db"],
                p=aug_cfg["bgnoise"]["p"],
            ),
            Gain(
                min_gain_in_db=aug_cfg["gain"]["min_gain_in_db"],
                max_gain_in_db=aug_cfg["gain"]["max_gain_in_db"],
                p=aug_cfg["gain"]["p"],
            ),
            HighPassFilter(
                min_cutoff_freq=aug_cfg["hipass"]["min_cutoff_freq"],
                max_cutoff_freq=aug_cfg["hipass"]["max_cutoff_freq"],
                # min_rolloff=6,
                # max_rolloff=24,
                p=1,
            ),
            LowPassFilter(
                min_cutoff_freq=aug_cfg["lopass"]["min_cutoff_freq"],
                max_cutoff_freq=aug_cfg["lopass"]["max_cutoff_freq"],
                p=aug_cfg["lopass"]["p"],
                # min_rolloff=6,
                # max_rolloff=24,
            ),
            ApplyImpulseResponse(
                ir_paths=[aug_cfg["reverb"]["path"]],
                p=aug_cfg["reverb"]["p"],
                # leave_length_unchanged=True,
            ),
        ]
    )
    return augmentations_callable
