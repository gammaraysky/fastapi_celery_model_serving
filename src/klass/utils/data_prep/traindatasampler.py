"""
For sampling from the primary (cleaned) dataset.

Run in 2 stages, the first stage chunks all train data into chunks of
fixed duration. The second stage is given total duration and proportions
of each dataset to sample from, and will randomly sample from the chunks
to produce a new train data mixture set.
"""
import hashlib
import logging
import math
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Set, Union

# import hydra
import librosa
import numpy as np
import soundfile as sf

from . import datavalidation as dval

logger = logging.getLogger(__name__)


class TrainDataSampler:
    """Handles chunking and subsampling from train datasets in order to
    create train data mixtures.
    """

    def check_number_of_keys(self, expected_no_of_keys: int, no_of_keys: int) -> bool:
        """
        Function checks that number of keys are similar.
        Args:
            expected_no_of_keys (int): Length of list of expected keys
            no_of_keys (int) : Length of list of keys

        Returns:
            result (bool) : True if passed, else False
        """
        if expected_no_of_keys != no_of_keys:
            return False

        else:
            return True

    def check_key_naming_consistency(
        self, expected_key_names: Set, key_names: Set
    ) -> bool:
        """
        Function checks that keys are named similarly.
        Args:
            expected_key_names (set): set of expected key names
            key_names (set): set of key names

        Returns:
            result (bool) : True if passed, else False
        """
        if expected_key_names != key_names:
            return False

        else:
            return True

    def check_for_duplicate_keys(
        self, list_of_key_names: List, set_of_key_names: Set
    ) -> bool:
        """
        Function checks that keys are named similarly.
        Args:
            list_of_key_names (list): List of keys
            set_of_key_names (set): Set of keys

        Returns:
            result (bool) : True if passed, else False
        """
        if len(list_of_key_names) != len(set_of_key_names):
            return False

        else:
            return True

    def check_for_divisibility(self, dividend: float, divisor: int) -> bool:
        """
        Function checks that the dividend is completely divisible by the divisor.
        Args:
            dividend (float): Expected input is the total_training_mins converted to secs
            divisor (int): Expected input is the audio_file_chunk_size in secs

        Returns:
            result (bool) : True if passed, else False
        """
        if (float(dividend) % int(divisor)) != 0:
            return False

        else:
            return True

    def config_file_check(self, cfg: Dict) -> bool:
        """
        Function does validation of inputs into configuration file
        for the specific use case of chunking and sampling.
        Args:
            cfg (DictConfig): Hydra's configuration object instance

        Returns:
            result (bool) : True if passed all tests, else False
        """

        # cfg['data_paths']['traindatasampler']['dataset_audio_path']
        # cfg['data_paths']['traindatasampler']['dataset_rttm_path']
        # cfg['data_paths']['traindatasampler']['chunked_output_path_audio']
        # cfg['data_paths']['traindatasampler']['chunked_output_path_rttm']
        # cfg['data_paths']['traindatasampler']['sampled_output_path_audio']
        # cfg['data_paths']['traindatasampler']['sampled_output_path_rttm']

        configuration_passed = True

        dataset_names = list(
            cfg["data_paths"]["traindatasampler"]["dataset_audio_path"].keys()
        )
        set_of_dataset_names = set(
            cfg["data_paths"]["traindatasampler"]["dataset_audio_path"].keys()
        )

        # Check for consistency in dataset naming and number of datasets within configuration file
        for section in cfg["data_paths"]["traindatasampler"].keys():
            name_result = self.check_key_naming_consistency(
                set_of_dataset_names,
                set(cfg["data_paths"]["traindatasampler"][section].keys()),
            )
            if name_result is False:
                logger.error(
                    "Dataset names in %s are not consistent with configuration file",
                    section,
                )
                configuration_passed = False

            number_result = self.check_number_of_keys(
                len(dataset_names),
                len(cfg["data_paths"]["traindatasampler"][section].keys()),
            )
            if number_result is False:
                logger.error(
                    "Number of dataset in %s are not consistent with configuration file",
                    section,
                )
                configuration_passed = False

        if (
            self.check_number_of_keys(
                len(dataset_names),
                len(cfg["traindatasampler"]["dataset_proportions"].keys()),
            )
            is False
            or self.check_key_naming_consistency(
                set_of_dataset_names,
                set(cfg["traindatasampler"]["dataset_proportions"].keys()),
            )
            is False
        ):
            logger.error(
                "Number of dataset(s) or Naming of dataset(s) are not consistent between"
                " all_paths and parameters"
            )
            configuration_passed = False

        # Check for duplicates
        for section in cfg["data_paths"]["traindatasampler"].keys():
            duplicate_result = self.check_for_duplicate_keys(
                cfg["data_paths"]["traindatasampler"][section].keys(),
                set(cfg["data_paths"]["traindatasampler"][section].keys()),
            )
            if duplicate_result is False:
                logger.error("Duplicate keys identified in %s", section)
                configuration_passed = False

        if (
            self.check_for_duplicate_keys(
                cfg["traindatasampler"]["dataset_proportions"].keys(),
                set(cfg["traindatasampler"]["dataset_proportions"].keys()),
            )
            is False
        ):
            logger.error("Duplicate keys identified in dataset_proportions")
            configuration_passed = False

        try:
            if (
                float(cfg["traindatasampler"]["total_training_mins"]) <= 5.0
                or int(cfg["traindatasampler"]["audio_file_chunk_size_secs"]) < 1
            ):  # 5 minutes and 1 seconds are arbitrary
                logger.error(
                    "At the minimum, more than 5 minutes must be used for training and"
                    " audio_file_chunk_size more than 1 second"
                )
                configuration_passed = False

            if int(cfg["traindatasampler"].audio_file_sample_rate_hz) <= 0:
                logger.error("audio_file_sample_rate_hz must be larger than zero")
                configuration_passed = False

            if (
                self.check_for_divisibility(
                    float(cfg["traindatasampler"]["total_training_mins"]) * 60,
                    int(cfg["traindatasampler"]["audio_file_chunk_size_secs"]),
                )
                is False
            ):
                logger.error(
                    "total_training_mins must be divisible by audio_file_chunk_size_secs"
                )
                configuration_passed = False

        except ValueError:
            logger.error(
                "Inputs for total_training_mins, audio_file_chunk_size_secs and"
                " audio_file_sample_rate_hz should be numeric"
            )
            configuration_passed = False
            return configuration_passed

        else:
            try:
                if sum(cfg["traindatasampler"].dataset_proportions.values()) != 1:
                    logger.error("Proportions in dataset_proportions do not add to 1")
                    configuration_passed = False

            except ValueError:
                logger.error("Values in dataset_proportions should be floats")
                configuration_passed = False
                return configuration_passed

            return configuration_passed

    def sampler_inputs_check(
        self,
        proportions_per_dataset: dict,
        input_dataset_path: dict,
        output_dataset_path: dict,
        input_rttm_path: dict,
        output_rttm_path: dict,
        total_training_mins: float,
        audio_file_chunk_size_secs: int,
        sample_rate: int,
    ) -> None:
        """Validates sampler arguments before executing sampling.
        Quits the program if error is found.

        Args:
            proportions_per_dataset (dict): Dict with dataset names as
                keys and float value between 0 - 1 for the proportion
                for that dataset
            input_dataset_path (dict): _description_
            output_dataset_path (dict): _description_
            input_rttm_path (dict): _description_
            output_rttm_path (dict): _description_
            total_training_mins (float): _description_
            audio_file_chunk_size_secs (int): _description_
            sample_rate (int): _description_


        """
        dataset_names = list(proportions_per_dataset.keys())
        set_of_dataset_names = set(proportions_per_dataset.keys())

        if self.check_for_duplicate_keys(dataset_names, set_of_dataset_names) is False:
            logger.error("Duplicate keys identified in proportions_per_dataset")
            sys.exit()

        for arguments in [
            input_dataset_path,
            output_dataset_path,
            input_rttm_path,
            output_rttm_path,
        ]:
            name_result = self.check_key_naming_consistency(
                set_of_dataset_names, set(arguments.keys())
            )
            if name_result is False:
                logger.error(
                    "Dataset names in %s are not consistent with inputs for proportions_per_dataset",
                    arguments,
                )
                sys.exit()

            number_result = self.check_number_of_keys(
                len(dataset_names), len(arguments.keys())
            )
            if number_result is False:
                logger.error(
                    "Number of dataset in %s are not consistent with inputs for proportions_per_dataset",
                    arguments,
                )
                sys.exit()

            duplicate_result = self.check_for_duplicate_keys(
                arguments.keys(), set(arguments.keys())
            )
            if duplicate_result is False:
                logger.error("Duplicate keys identified in %s", arguments)
                sys.exit()

        try:
            if sum(proportions_per_dataset.values()) != 1:
                logger.error(
                    "Proportions given in proportions_per_dataset do not sum to 1"
                )
                sys.exit()

            if sample_rate <= 0:
                logger.error("audio_file_sample_rate_hz must be larger than zero")
                sys.exit()

            if (
                self.check_for_divisibility(
                    float(total_training_mins) * 60, int(audio_file_chunk_size_secs)
                )
                is False
            ):
                logger.error(
                    "total_training_mins (when converted to secs) must "
                    "be divisible by audio_file_chunk_size_secs"
                )
                sys.exit()

        except ValueError:
            logger.error(
                "proportions_per_dataset, sample_rate, "
                "total_training_mins and audio_file_chunk_size_secs "
                "should be numeric"
            )
            sys.exit()

    def signal_array_clipper(
        self, signal_array: np.ndarray, sample_rate: int, chunk_size: int
    ) -> np.ndarray:
        """
        Function does validation of inputs into configuration file
        for the specific use case of chunking and sampling.
        Args:
            signal_array (numpy.array) : samples from audio file in
                array form
            sample_rate (int) : sample rate of audio file
            chunk_size (int) : audio chunk size required in seconds

        Returns:
            signal_array (numpy.array) : samples from audio file in
            array form, clipped to the nearest chunk_size multiple
        """
        return signal_array[
            0 : (len(signal_array) // (chunk_size * sample_rate))
            * (chunk_size * sample_rate)
            + 1
        ]

    def create_directory(self, directory_path: str) -> None:
        """
        TODO:Potentially able to consolidate this function!!
        Function creates a directory if it does not exists
        Args:
            directory_path (str): directory path in string

        """
        path = Path(directory_path)
        if not path.exists():
            path.mkdir(parents=True)
            logger.info("Directory '%s' created successfully.", path)
        else:
            logger.warning(
                "Directory '%s' already exists, folder may contain unexpected rttm/wav files.",
                path,
            )

    def wav_folder_path_reader(self, wav_folder_path: str) -> Set:
        """
        TODO:Potentially able to consolidate this function!!
        Function returns a set of .wav audio path given a path directory
        Args:
            wav_folder_path (str): directory path in string

        Returns:
            set(list_of_audio_path) (set) : set of .wav audio path
        """
        list_of_audio_path = []

        for path in Path(wav_folder_path).rglob("*.wav"):
            list_of_audio_path.append(path)

        return set(list_of_audio_path)

    def audio_file_loader(self, audio_path: str, sample_rate: int) -> np.ndarray:
        """
        Function converts an audio file into
        a signal array given an audio file path and a sample rate
        Args:
            audio_path (str): audio file path in string format
            sample_rate (int) : sample rate of audio file

        Returns:
            signal_array (numpy.array) : samples from audio file in
                array form
        """
        try:
            signal_array, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

        except Exception:
            logger.warning(traceback.format_exc())

        return signal_array

    def signal_array_chunker(
        self, signal_array: np.ndarray, sample_rate: int = 16000, chunk_size: int = 300
    ):
        """
        Function returns a dictionary of signal chunks in blocks of
        chunk_size (seconds).
        Default chunk_size = 300 seconds and sample_rate = 16kHz.
        Any remaider signal not within a chunk_size will be discarded
        Args:
            signal_array (numpy.array) : samples from audio file in
                array form
            sample_rate (int) : sample rate of audio
            chunk_size (int) : audio chunk size required in seconds

        Returns:
            signal_chunks_dict (dict) : Dictionary containing audio file
                name as keys and signal array of chunk size as values
        """
        signal_chunks_dict = {}

        if len(signal_array) <= 0:
            return signal_chunks_dict

        if len(signal_array) % (chunk_size * sample_rate) != 0:
            signal_array = self.signal_array_clipper(
                signal_array, sample_rate, chunk_size
            )

        for i in range(len(signal_array) // (chunk_size * sample_rate)):
            # Naming the keys of signal_chunks_dict based on the chunk it represents
            signal_chunks_dict[
                "{}-{}".format(i * chunk_size, (i + 1) * chunk_size)
            ] = signal_array[
                i * chunk_size * sample_rate : ((i + 1) * chunk_size * sample_rate) + 1
            ]

        return signal_chunks_dict

    def generate_chunked_rttm(
        self, input_rttm_path: str, out_rttm_path: str, start_time: int, end_time: int
    ) -> None:
        """
        Generates a .rttm file by extracting a portion of a full .rttm
        file given a start and end time interval
        Args:
            input_rttm_path (str) : input path of rttm file
            out_rttm_path (str) : output path of rttm file
            start_time (int) : Portion of rttm file to start extraction from
            end_time (int) : Portion of rttm file to extract  until

        """
        rttm_subset = []
        with open(input_rttm_path, "r") as full_rttm:
            list_of_lines = full_rttm.readlines()
            logger.debug("%s read successfully", input_rttm_path)

        for lines in list_of_lines:
            list_of_columns = lines.split()

            filename = (
                os.path.basename(input_rttm_path).split(".rttm")[0]
                + "_"
                + "{}".format(start_time)
                + "_"
                + "{}".format(end_time)
            )

            # if not within criteria (chunk timing for this rttm)
            if (
                float(list_of_columns[3]) + float(list_of_columns[4]) < start_time
                or float(list_of_columns[3]) > end_time
            ):
                continue
            else:
                list_of_columns[3] = float(list_of_columns[3]) - start_time
                if float(list_of_columns[3]) < 0:
                    # Correction for cases where the speech starts before
                    # start_time but ends within start_time and end_time
                    list_of_columns[4] = float(list_of_columns[4]) + float(
                        list_of_columns[3]
                    )
                    list_of_columns[3] = 0

                if (float(list_of_columns[3]) + float(list_of_columns[4])) > (
                    end_time - start_time
                ):
                    # Correction for cases where the speech starts during
                    # start_time and end_time but ends after end_time
                    list_of_columns[4] = (end_time - start_time) - float(
                        list_of_columns[3]
                    )

                list_of_columns[1] = filename

                # if duration > 0, append row
                if float(list_of_columns[4]) > 0:
                    rttm_subset.append(list_of_columns)

        if rttm_subset != []:
            with open(
                "{}/{}.rttm".format(out_rttm_path, filename), "w"
            ) as chunked_rttm:
                for lines in rttm_subset:
                    for columns in lines:
                        chunked_rttm.write(str(columns))
                        chunked_rttm.write(" ")
                    chunked_rttm.write("\n")
                logger.debug("%s generated successfully", out_rttm_path)

        else:
            filename = (
                os.path.basename(input_rttm_path).split(".rttm")[0]
                + "_"
                + "{}".format(start_time)
                + "_"
                + "{}".format(end_time)
            )
            with open(
                "{}/{}.rttm".format(out_rttm_path, filename), "w"
            ) as chunked_rttm:
                chunked_rttm.write("")
                logger.warning("Empty .rttm file created for %s", filename)

    def audio_file_name_hasher(self, filename):
        """
        Function converts a filename given in string format into a
            Secure Hash Algorithm 256-bit hexadecimal string
        Args:
            filename (str) : Audio file name in string
        Returns:
            sha256_hash.hexdigest() (str) : string object in hexadecimal
                digits
        """
        sha256_hash = hashlib.sha256()
        sha256_hash.update(filename.encode("utf-8"))
        logger.debug("%s successfully hashed", filename)

        return sha256_hash.hexdigest()

    def output_validation(
        self, list_of_audio_file_directory, dict_of_rttm_file_directory
    ) -> None:
        """
        Function iterates through audio file directory and rttm file
        directory and finds the differences
        Args:
            list_of_audio_file_directory (list) : List with element 0
                containing dataset name and element 1 containing audio
                file path
            dict_of_rttm_file_directory (dict) : dictionary with keys
                referencing to dataset name and value referencing rttm
                path
        """
        list_of_audio = []
        list_of_rttm = []
        audio_file_path = list_of_audio_file_directory[1]
        output_dataset_name = list_of_audio_file_directory[0]

        for path in Path(audio_file_path).rglob("*.wav"):
            audio_file_name = os.path.basename(path).split(".wav")[0]
            list_of_audio.append(audio_file_name)

        for path in Path(dict_of_rttm_file_directory[output_dataset_name]).rglob(
            "*.rttm"
        ):
            rttm_file_name = os.path.basename(path).split(".rttm")[0]
            list_of_rttm.append(rttm_file_name)

        if len(list_of_rttm) != len(list_of_audio) or set(list_of_rttm) != set(
            list_of_audio
        ):
            logger.warning(
                ".rttm files do not match chunked .wav files either in number or naming"
            )
            logger.warning(
                "Identified differences in naming %s",
                set(list_of_rttm).difference(set(list_of_audio)),
            )

            if len(list_of_rttm) > len(list_of_audio):
                differences = [x for x in list_of_rttm if x not in list_of_audio]
                logger.warning(
                    "List of .rttm files exists, but not .wav files %s", differences
                )

            else:
                differences = [x for x in list_of_audio if x not in list_of_rttm]
                logger.warning(
                    "List of .wav files exists, but not .rttm files %s", differences
                )

    def copy_audio_and_rttm_file(
        self,
        dict_item,
        dict_input_audio_path,
        dict_output_audio_path,
        dict_input_rttm_path,
        dict_output_rttm_path,
    ) -> None:
        """
        Function copies audio and rttm file, given an dictionary of
        input and output audio and rttm paths
        Args:
            dict_item (dict) : List with element 0 containing dataset
                name and element 1 containing a list of dictionary with
                keys referencing audio file name hashes and values
                representing the filename
            dict_input_audio_path (dict) : dictionary with keys
                referencing to dataset name and value referencing input
                audio path
            dict_output_audio_path (dict) : dictionary with keys
                referencing to dataset name and value referencing input
                audio path
            dict_input_rttm_path (dict) : dictionary with keys
                referencing to dataset name and value referencing input
                rttm path
            dict_output_rttm_path (dict) : dictionary with keys
                referencing to dataset name and value referencing output
                rttm path
        """
        dataset_name = dict_item[0]
        list_of_dict_of_selected_file_names = dict_item[1]

        for filename in list_of_dict_of_selected_file_names:
            shutil.copy2(
                dict_input_audio_path[dataset_name]
                + "/{}.wav".format(list(filename.values())[0]),
                dict_output_audio_path[dataset_name]
                + "/{}.wav".format(list(filename.values())[0]),
            )

            shutil.copy2(
                dict_input_rttm_path[dataset_name]
                + "/{}.rttm".format(list(filename.values())[0]),
                dict_output_rttm_path[dataset_name]
                + "/{}.rttm".format(list(filename.values())[0]),
            )

    def audio_file_sampler(
        self,
        proportions_per_dataset: dict,
        input_dataset_path: dict,
        output_dataset_path: dict,
        input_rttm_path: dict,
        output_rttm_path: dict,
        total_training_mins: float,
        audio_file_chunk_size_secs: int,
        sample_rate: int,
    ):
        """
        Function picks samples of audio files from a given value in
        input_dataset_path and copies these files into a given value in
        output_dataset_path.

        Args:
            proportions_per_dataset (dict): Dictionary with key
                referencing dataset name and value referencing the
                proportion
            input_dataset_path (dict): Dictionary with key referencing
                dataset name and value referencing dataset path
            output_dataset_path (dict): Dictionary with key referencing
                dataset name and value referencing dataset path
            input_rttm_path (dict): Dictionary with key referencing
                dataset name and value referencing dataset path
            output_rttm_path (dict): Dictionary with key referencing
                dataset name and value referencing dataset path
            total_training_mins (float) : Total amount of training
                minutes required
            audio_file_chunk_size_secs (int) : Audio file chunk size in
                secs of audio files in input_dataset_path
            sample_rate (int) : sample rate of audio files in
                input_dataset_path
        Returns:
            NA
        """

        # Input parameters check
        self.sampler_inputs_check(
            proportions_per_dataset,
            input_dataset_path,
            output_dataset_path,
            input_rttm_path,
            output_rttm_path,
            total_training_mins,
            audio_file_chunk_size_secs,
            sample_rate,
        )

        dataset_total_duration_min = {}
        dict_of_set_of_audio_path = {}

        validator = dval.TrainDataSamplerValidation()

        # VALIDATE FILE CHUNK LENGTHS CORRECT, CALCULATE TOTAL DURATION
        # MINS PER DATASET
        (
            error_datasets,
            dict_of_set_of_audio_path,
            dataset_total_duration_min,
        ) = validator.check_chunk_lengths(
            input_dataset_path,
            audio_file_chunk_size_secs,
            return_total_duration_mins=True,
        )

        # CHECK SUFFICIENT DURATION IN EACH DATASET TO PRODUCE REQUIRED
        # SAMPLING
        (
            results,
            error_datasets,
            duration_mins_per_dataset,
        ) = validator.check_sufficient_minutes(
            proportions_per_dataset,
            audio_file_chunk_size_secs,
            dataset_total_duration_min,
            total_training_mins,
        )
        if results is False:
            logger.error(
                "The following datasets do not have sufficient training minutes: %s",
                error_datasets,
            )
            sys.exit()

        # Hashing each file name in dataset (purpose is to pick a fixed
        # set of audio file in a reproducible manner)
        dict_of_hashed_name = self.generate_hash_for_dataset(dict_of_set_of_audio_path)

        # Pick out files from each dataset based on the number of training mins required
        dict_of_selected_files = {}
        for dataset_name, training_mins in duration_mins_per_dataset.items():
            total_files_required = training_mins / (
                int(audio_file_chunk_size_secs) / 60
            )
            dict_of_selected_files[dataset_name] = dict_of_hashed_name[dataset_name][
                0 : int(total_files_required)
            ]

        # Copying each selected file into the appropriate output paths given
        for item in dict_of_selected_files.items():
            self.copy_audio_and_rttm_file(
                item,
                input_dataset_path,
                output_dataset_path,
                input_rttm_path,
                output_rttm_path,
            )

        for item in output_dataset_path.items():
            self.output_validation(list(item), output_rttm_path)

        logger.info(
            "Sampling Completed, total training minutes used per dataset : %s \nTotal"
            " training minutes sampled : %s",
            list(duration_mins_per_dataset.items()),
            sum(duration_mins_per_dataset.values()),
        )

    def generate_hash_for_dataset(self, dict_of_set_of_audio_path: Dict) -> Dict:
        # Hashing each file name in dataset (purpose is to pick a fixed
        # set of audio file in a reproducible manner)
        dict_of_hashed_name = {}
        for dataset_name, set_of_audio_paths in dict_of_set_of_audio_path.items():
            for path in set_of_audio_paths:
                filename = os.path.basename(path).split(".wav")[
                    0
                ]  # To get the audio file name
                hashed_name = self.audio_file_name_hasher(filename)
                if dataset_name not in dict_of_hashed_name:
                    dict_of_hashed_name[dataset_name] = [{hashed_name: filename}]
                else:
                    dict_of_hashed_name[dataset_name].append({hashed_name: filename})

        # Sort the dict_of_hashed_name in order of hashkeys
        for item in dict_of_hashed_name.items():
            dict_of_hashed_name[item[0]] = sorted(
                dict_of_hashed_name[item[0]], key=lambda x: list(x.keys())[0]
            )

        return dict_of_hashed_name
