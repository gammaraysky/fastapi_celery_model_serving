"""datavalidation.py

Collection of methods and class to conduct data validation on audio
and rttm files.

"""
import hashlib
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union

import librosa
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)


def copy_files_recursive(source, destination):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination, exist_ok=True)

    # Iterate over the contents of the source folder
    for item in os.listdir(source):
        item_path = os.path.join(source, item)
        dest_path = os.path.join(destination, item)

        if os.path.isdir(item_path):
            # If it's a directory, recursively call the function
            copy_files_recursive(item_path, dest_path)
        else:
            # If it's a file, copy it to the destination
            if os.path.isfile(dest_path):
                logger.warning("%s exists, did not copy.", str(dest_path))
            else:
                shutil.copy2(item_path, dest_path)


def drop_irrelevant_folders(basepath):
    """Expects given basepath to contain the following folders:
    basepath/train/audio
            /train/rttm
            /val/audio
            /val/rttm
            /test/audio
            /test/rttm

    any other folders or files will be removed.

    """
    report = {}
    for split in ["train", "val", "test"]:
        report[split] = {}

        # get folder contents in each train/val/test subfolder.
        folder_contents = list(os.listdir(Path(basepath).joinpath(split)))
        for item in folder_contents:
            # if items are not audio and rttm folders, delete them
            if str(item) != "rttm" and str(item) != "audio":
                itempath = Path(basepath).joinpath(split).joinpath(item)
                if itempath.is_dir():
                    report[split][item] = "dir"
                    shutil.rmtree(itempath)
                    logger.info("Deleting folder: %s", itempath)
                elif itempath.is_file():
                    report[split][item] = "file"
                    os.remove(itempath)
                    logger.info("Deleting file: %s", itempath)
    return report


class DataValidation:
    """DataValidation class which contains a collection of data
    validation methods.
    """

    def __init__(self, dataset: Dict, directory: Union[Path, str]):
        self.dataset = dataset
        self.directory = Path(directory)

    def check_audio_rttm_mismatches(self) -> Dict:
        """Checks through 'train', 'val', and 'test' splits and finds
        audio or rttm files for which they are missing their counterpart
        audio/rttm pairs.
        e.g. file001.wav should have corresponding file001.rttm

        Returns:
            Dict: a dict with following output:
                {
                    'train' : [
                        Path('path/to/file01.wav'),
                        Path('path/to/file02.rttm'),
                    ],
                    'val' : [ ... ],
                    'test' : [ ... ],
                }
        """ """
        """
        dangling_files = {}
        for split in ["train", "val", "test"]:
            waves = list(self.directory.joinpath(split).joinpath("audio").glob("*.wav"))
            rttms = list(self.directory.joinpath(split).joinpath("rttm").glob("*.rttm"))

            dangling_files[split] = self._get_unmatched_audio_rttm_pairs(waves, rttms)

        return dangling_files

    def delete_audio_rttm_mismatches(self, dangling_files: Dict) -> None:
        """Deletes dangling audio/rttm files without matching pairs

        Args:
            dangling_files (Dict[Dict[List]]): Matches dict returned by
                self.check_audio_rttm_mismatches()

        """ """"""
        for split in ["train", "val", "test"]:
            for path in dangling_files[split]:
                try:
                    os.remove(path)
                    logger.info("Deleted: %s ", path)
                except OSError as e:
                    logger.error("Unable to delete %s : %s", path, e)
                try:
                    file_id = Path(path).stem
                    del self.dataset[split][file_id]
                except Exception as e:
                    logger.error("File_id not found in self.dataset : %s", e)

    def _get_unmatched_audio_rttm_pairs(
        self, list_a: List[Path], list_b: List[Path]
    ) -> List[Path]:
        """Compares 2 lists of filepaths and uses hashmap to find
        dangling files i.e no matching pair e.g. file1.wav and
        file1.rttm.

        Args:
            list_a (List[Path]): list of audio file paths
            list_b (List[Path]): list of rttm file paths

        Returns:
            List[Path]: list of file stems for which no matching pair
                found in both lists.
        """
        final = {}

        for path in list_a + list_b:
            key = path.stem
            if key in final:
                final.pop(key)
            else:
                final[key] = path

        return list(final.values())

    def format_check(self) -> Dict:
        """Given a dataset Dict, scans all audio files and returns
        dictionaries for results where an audio file is:
        - not a .wav file
        - not mono channeled
        - not 16KHz sample rate

        Returns:
            report - a Dict containing the following keys:
                non_wave_files
                non_mono_files
                non_16khz_files

                each are dictionaries reporting affected files:
                key -       dataset subclass (e.g. ami_train_far)
                value -     List of filenames affected (e.g.
                            EN2002a.Headset-1.wav) (where the above
                            conditions are detected)

        """
        non_wave_files = {}
        non_mono_files = {}
        non_16khz_files = {}

        # loop through train val test
        for split in self.dataset.keys():
            non_wave_files[split] = []
            non_mono_files[split] = []
            non_16khz_files[split] = []

            # loop through each file_id in subset
            for file_id in tqdm(sorted(self.dataset[split].keys())):
                audio_path = self.dataset[split][file_id]["audio_path"]

                # if not wave file
                if Path(audio_path).suffix != ".wav":
                    non_wave_files[split].append(file_id)

                # if wave file
                else:
                    info = sf.info(audio_path)
                    target_sample_rate = 16000

                    # if not mono
                    if info.channels > 1:
                        non_mono_files[split].append(file_id)

                    # if not 16khz
                    if info.samplerate != target_sample_rate:
                        non_16khz_files[split].append(file_id)

            non_wave_files[split] = sorted(non_wave_files[split])
            non_mono_files[split] = sorted(non_mono_files[split])
            non_16khz_files[split] = sorted(non_16khz_files[split])
            # end loop through each file_id in subset

        # end loop through train val test

        format_check_report = {
            "non_wave": non_wave_files,
            "non_mono": non_mono_files,
            "non_16khz": non_16khz_files,
        }

        return format_check_report

    def format_convert_mono(self, report: Dict) -> None:
        """Given a report Dict returned by self.format_check(), converts
        the files flagged as not mono channeled and converts them to
        mono and overwrites the original file.

        Args:
            report (Dict): matches Dict format outputted by
                self.format_check()
        """
        for issue in ["non_mono"]:
            for split in ["train", "val", "test"]:
                file_ids = report[issue][split]
                for file_id in file_ids:
                    self._save_to_mono(self.dataset[split][file_id]["audio_path"])

    @staticmethod
    def _save_to_mono(audio_path: Union[str, Path]) -> None:
        """Reads in the audio file at audio_path, takes only the
        leftmost channel, and overwrites it as a mono file.

        Args:
            audio_path (Union[str, Path]): path to stereo/multi-channel
                audio file
        """

        try:
            signal, samplerate = sf.read(audio_path)
            leftmost_signal = signal[:, 0]

        except Exception as e:
            logger.error("Cannot read audio %s : %s", audio_path, e)
            return

        try:
            sf.write(audio_path, leftmost_signal, samplerate, subtype="PCM_16")
        except Exception as e:
            logger.error("Cannot save audio %s : %s", audio_path, e)
            return

    def format_convert_16khz(self, report: Dict) -> None:
        """Given a report Dict returned by self.format_check(), converts
        the files flagged as not 16KHz sample rate and converts them to
        16KHz and overwrites the original file.

        Args:
            report (Dict): matches Dict format outputted by
                self.format_check()
        """
        for issue in ["non_16khz"]:
            for split in ["train", "val", "test"]:
                file_ids = report[issue][split]
                for file_id in file_ids:
                    self._save_to_16khz(self.dataset[split][file_id]["audio_path"])

    @staticmethod
    def _save_to_16khz(audio_path: Union[Path, str]) -> None:
        """Reads in the audio file at audio_path, resamples signal to
        16KHz, and overwrites it as a mono file.

        Args:
            audio_path (Union[str, Path]): path to non-16KHz audio file
        """
        try:
            signal, samplerate = sf.read(audio_path)
            target_sr = 16000
            signal_resamp = librosa.resample(signal, samplerate, target_sr)
        except Exception as e:
            logger.error("Error during resampling %s : %s", audio_path, e)
            return
        try:
            sf.write(audio_path, signal_resamp, target_sr)
        except Exception as e:
            logger.error("Cannot save audio %s : %s", audio_path, e)
            return

    def format_drop_nonwave(self, report: Dict) -> None:
        """Given a report Dict returned by self.format_check(), converts
        the files flagged as not wave files, deletes the file from disk
        and drops them from the dataset.

        Args:
            report (Dict): matches Dict format outputted by
                self.format_check()
        """
        for issue in ["non_wave"]:
            for split in ["train", "val", "test"]:
                file_ids = report[issue][split]
                for file_id in file_ids:
                    self._remove_entry(split, file_id)

    def annot_check(self) -> Tuple:
        """Checks loaded annotations for errors:
            - annotations empty, no speech segments
            - annotations too short: less than half of audio duration
            - annotations too long: longer than audio duration
            - annotations time errors: start time later than end time

        Returns:
            a Dict as follows:
            {
                'train' : {
                    '<file_id>' : '<error type>',
                    '<file_id>' : '<error type>',
                }
                'val' : {}
                'test' : {}
            }
        """
        annot_errors = {}

        # train val test split
        for split in self.dataset.keys():
            annot_errors[split] = {}
            for file_id in tqdm(sorted(self.dataset[split].keys())):
                (
                    is_annot_empty,
                    is_annot_short,
                    is_annot_long,
                    is_annot_time_mismatch,
                ) = (None, None, None, None)

                audio_path = self.dataset[split][file_id]["audio_path"]
                segments = self.dataset[split][file_id]["segments"]

                if len(segments) == 0:
                    is_annot_empty = True

                else:
                    duration = self.get_wave_duration(str(audio_path))
                    last_segment_end = segments[-1][-1]
                    diff = last_segment_end - duration

                    # give 1s allowance to allow for some discrepancy in
                    # audio file duration vs last segment ending
                    if last_segment_end > duration + 1:
                        is_annot_long = True

                    if last_segment_end * 2 < duration:
                        is_annot_short = True

                    for start, end in segments:
                        if start > end:
                            is_annot_time_mismatch = True

                if (
                    is_annot_empty
                    or is_annot_short
                    or is_annot_long
                    or is_annot_time_mismatch
                ):
                    err_msg = ""
                    err_msg += "annot empty" if is_annot_empty else ""
                    err_msg += "annot short" if is_annot_short else ""
                    err_msg += f"annot long {diff:.2f}s" if is_annot_long else ""
                    err_msg += "annot mismatch" if is_annot_time_mismatch else ""

                    annot_errors[split][file_id] = err_msg

        return annot_errors

    def annot_drop_erroneous(self, report: Dict) -> None:
        """Given a report Dict returned by self.annot_check(), looks up
        files that were flagged as having annotations errors, and
        deletes the corresponding audio file and rttm file pair from
        disk, and drops them from the dataset.

        Args:
            report (Dict): matches Dict format outputted by
                self.annot_check()
        """
        splits = list(report.keys())
        for split in splits:
            file_ids = list(report[split].keys())
            for file_id in file_ids:
                self._remove_entry(split, file_id)

    def _remove_entry(self, split, file_id) -> None:
        """Deletes both audio file and rttm file as denoted in
        audio_path and rttm_path keys, as well as drops the fileid key
        from the self.dataset dict.
        """
        # delete audio file, rttm file, delete dict key
        audio_file = self.dataset[split][file_id]["audio_path"]
        rttm_file = self.dataset[split][file_id]["rttm_path"]

        self._remove_file(audio_file)
        self._remove_file(rttm_file)
        del self.dataset[split][file_id]

    @staticmethod
    def _remove_file(filepath):
        """Delete file from disk, or logs an error if unable to delete."""
        try:
            os.remove(filepath)
            logger.debug(f"{filepath} successfully deleted.")
        except OSError as e:
            logger.error(f"Error occurred while deleting {filepath}: {e}")

    def get_wave_duration(self, file_path) -> float:
        """Retrieve a audio file's duration in seconds"""
        info = sf.info(file_path)
        return info.duration

    def get_file_hash(self, file_path) -> str:
        """Calculates and returns the hash of a file's content.
        Modified for speed, only loads the first 10240 bytes for
        calculating hash on.
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as file:
            chunk = file.read(10240)
            hasher.update(chunk)

        return hasher.hexdigest()

    def find_duplicate_files(self) -> List[Tuple[str, str]]:
        """Finds duplicate files within a directory.

        Returns: List of duplicates, as a tuple of
            (file_path, hash_string)
        """
        file_hashes = {}
        duplicates = []

        # Traverse through all files in the directory and its subdirectories
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_hash = self.get_file_hash(file_path)

                # Check if the file hash already exists
                if file_hash in file_hashes:
                    duplicates.append((file_path, file_hashes[file_hash]))
                else:
                    file_hashes[file_hash] = file_path

        return duplicates


class TrainDataSamplerValidation:
    def __init__(self):
        pass

    def check_chunk_lengths(
        self,
        input_dataset_path: str,
        audio_file_chunk_size_secs: int,
        return_total_duration_mins=True,
    ) -> Tuple[Dict, Dict, Union[Dict, None]]:
        """Checks the input data path to ensure all wave files per
        dataset adhere to the `audio_file_chunk_size_secs` specified.

        Args:
            input_dataset_path (str): Path to input chunked wave files
            audio_file_chunk_size_secs (int): Audio file chunk duration
                in secs
            return_total_duration_mins (bool, optional): If True,
                returns total duration in mins as the last item in the
                tuple. Defaults to True.

        Returns:
            Tuple[Dict, Dict, Union[Dict, None]]: A Tuple of 2 or 3
                dicts, depending if `return_total_duration_mins` was
                True:
                error_datasets, dict_of_set_of_audio_path,
                dataset_total_duration_min

                Each dict has the dataset names as keys.

                `error_datasets` values are the list of filepaths that do
                    not match audio_file_chunk_size_secs, per dataset.
                `dict_of_set_of_audio_path` values are the list of
                    filepaths that match audio_file_chunk_size_secs, per
                    dataset.
                `dataset_total_duration_min` values are the total
                    duration in minutes for each dataset.
        """

        error_datasets = {}
        dict_of_set_of_audio_path = {}
        dataset_total_duration_min = {}

        # loop through multiple datasets
        for dataset_name, audio_folder in input_dataset_path.items():
            num_audio_files = 0
            errorfiles = []
            correctfiles = []
            set_of_audio_paths = list(set(list(Path(audio_folder).glob("*.wav"))))

            # loop through files in a dataset
            for audio_path in set_of_audio_paths:
                if not self.check_audio_duration_match(
                    audio_path, int(audio_file_chunk_size_secs)
                ):
                    logger.error(
                        "%s duration is not %s", audio_path, audio_file_chunk_size_secs
                    )
                    errorfiles.append(audio_path)
                else:
                    correctfiles.append(audio_path)
                    num_audio_files += 1

            duration_min = float(num_audio_files * int(audio_file_chunk_size_secs) / 60)

            # save to output dicts
            dict_of_set_of_audio_path[dataset_name] = correctfiles
            dataset_total_duration_min[dataset_name] = duration_min
            if len(errorfiles) > 0:
                error_datasets[dataset_name] = errorfiles

        if return_total_duration_mins:
            return error_datasets, dict_of_set_of_audio_path, dataset_total_duration_min

        return error_datasets, dict_of_set_of_audio_path

    def check_audio_duration_match(
        self, audio_path: Union[Path, str], audio_file_chunk_size_secs: int
    ) -> bool:
        duration = sf.info(audio_path).duration
        if int(duration) != int(audio_file_chunk_size_secs):
            return False
        return True

    def check_sufficient_minutes(
        self,
        proportions_per_dataset: Dict,
        audio_file_chunk_size_secs: int,
        dataset_total_duration_min: Dict,
        total_training_mins: int,
    ) -> Tuple[bool, List]:
        """Checks whether each dataset chunks has sufficient durations
        to fulfill the sampling proportion requirements.

        Args:
            proportions_per_dataset (Dict): Dict with dataset names as
                keys and float 0-1 proportion for its sampling
                proportion.
                        e.g. { 'ali_far' : 0.5,
                            'ami_far' : 0.5 }
            audio_file_chunk_size_secs (int): Duration of each audio
                file chunk in seconds
            dataset_total_duration_min (Dict): Dict with dataset names
                as keys and int value of total duration in mins that the
                dataset contains.

        Returns:
            Tuple[bool, List, Dict]: A tuple of:
                bool - True or False whether there is sufficient minutes
                List - Dataset names which have insufficient minutes,
                    None if they all have sufficient duration to sample
                Dict - dict with dataset name as keys and durations in
                    mins as values
        """

        duration_mins_per_dataset = {}
        for dataset_name, dataset_proportion in proportions_per_dataset.items():
            duration_mins_per_dataset[dataset_name] = self.mins_required_from_dataset(
                dataset_proportion, total_training_mins, audio_file_chunk_size_secs
            )

        # Check that there is sufficient training mins in each dataset
        # as required by mins_required_from_dataset
        error_datasets = []
        for dataset_name in duration_mins_per_dataset.keys():
            if (
                duration_mins_per_dataset[dataset_name]
                > dataset_total_duration_min[dataset_name]
            ):
                error_datasets.append(dataset_name)

        if len(error_datasets) > 0:
            return (False, error_datasets, duration_mins_per_dataset)

        return (True, error_datasets, duration_mins_per_dataset)

    def mins_required_from_dataset(
        self,
        dataset_proportion: float,
        total_training_mins: int,
        audio_file_chunk_size_secs: int,
    ) -> int:
        mins_required_from_dataset = float(dataset_proportion) * total_training_mins

        return self.round_nearest_chunk_size(
            mins_required_from_dataset, audio_file_chunk_size_secs
        )

    def round_nearest_chunk_size(
        self, mins_required_from_dataset, audio_file_chunk_size_secs
    ):
        """
        Function rounds a float bigger than or equal to 0.5 to the next
            biggest integer, and rounds down otherwise
        Args:
            mins_required_from_dataset (float) : total training
                minutes required
            audio_file_chunk_size_secs (int) : audio file size given in
                secs
        Returns:
            mins_required_from_dataset (float) : rounded
                mins_required_from_dataset
        """
        if (mins_required_from_dataset * 60) / int(
            audio_file_chunk_size_secs
        ) - math.floor(
            (mins_required_from_dataset * 60) / int(audio_file_chunk_size_secs)
        ) < 0.5:
            mins_required_from_dataset = (
                (mins_required_from_dataset * 60) // int(audio_file_chunk_size_secs)
            ) * (audio_file_chunk_size_secs / 60)

        else:
            mins_required_from_dataset = (
                math.ceil(
                    (mins_required_from_dataset * 60) / int(audio_file_chunk_size_secs)
                )
            ) * (audio_file_chunk_size_secs / 60)

        return mins_required_from_dataset
