"""prepdatafolders.py

Use to reorganize your original dataset folder structure and
speech annotations into a custom format that our data pipeline can
ingest.

Use `PrepDataFolders.find_files_in_folder_recursive()` to selectively
grab files of a certain type as a list and `PrepDataFolders.copyfiles()`
to copy them into a new folder.

Organize your folder structure as follows:
data/interim/<dataset>/<split>/audio
                              /rttm
                              /textgrid
                              /xml
For the annotations, provide it in any of the above formats, ensuring
that filename matches each audio file.

Use converters below to convert them to RTTM format, which is our
preferred format for ingest.

For far field audio which combines speaker segments from multiple
near field audio files, use .merge_xml() or .merge_overlap_segments() to
generate combined segments.

Once dataset is ready, it can be passed through DataValidation() class.

"""
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union  # Optional,

import numpy as np
import soundfile as sf
from tqdm import tqdm

from . import speech_segments as sseg

logger = logging.getLogger(__name__)


class PrepDataFolders:
    """
    PrepDataFolders contains generic methods such as
    `file_files_in_folder_recursive()` and `copyfiles()`
    to collate certain filetypes and copy over to a destination path,
    various methods for reading and writing to RTTM, XML, TextGrid
    files, etc.

    Also see AmiPrepDataFolders or AliPrepDataFolders, which extends
    PrepDataFolders for AMI/Alimeeting dataset-specific functionality.

    """

    def __init__(self):
        pass

    # ! TODO - pull out from here
    @staticmethod
    def toggle_audio_rttm_path(file_path: Union[Path, str]) -> Path:
        """Given a filepath to an audio file, returns the filepath for
        the corresponding rttm file, or vice versa.

        Args:
            file_path (Union[Path, str]): input file path

        Raises:
            ValueError: if file path is not to a .wav or .rttm file

        Returns:
            Path: path to the counterpart file pair.
        """
        if Path(file_path).suffix == ".wav":
            new_path = file_path.parent.parent.joinpath("rttm").joinpath(
                str(file_path.stem) + ".rttm"
            )
        elif Path(file_path).suffix == ".rttm":
            new_path = file_path.parent.parent.joinpath("audio").joinpath(
                str(file_path.stem) + ".wav"
            )
        else:
            raise ValueError(
                "File must be either .wav or .rttm file to toggle with the other."
            )
        return new_path

    @staticmethod
    def read_audio(audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Read in audio file from given file path.

        Args:
            audio_path (str|Path): Path to audio file

        Returns:
            Tuple[np.ndarray, int]: tuple of signal and sample rate
        """
        try:
            signal, sr = sf.read(str(audio_path))
            return signal, sr
        except Exception as e:
            logger.error("Unable to read file %s : %s", audio_path, e)
            return

    @staticmethod
    def save_audio(audio_path, signal, sr, subtype: str = "PCM_16") -> Union[None, int]:
        try:
            sf.write(audio_path, signal, sr, subtype=subtype)
            return
        except Exception as e:
            logger.error("Unable to write file %s : %s", audio_path, e)
            return -1

    def duplicate_rttm_file(self, rttm_path, rttm_copy_path) -> Union[None, int]:
        try:
            # copies file
            shutil.copy2(rttm_path, rttm_copy_path)
            # edits file and replaces 'fileid's in each segment to
            # reflect new filename
            self.update_rttm_fileid(rttm_copy_path)
        except Exception as e:
            logger.error(
                "Unable to copy or create new RTTM %s : %s",
                rttm_copy_path,
                e,
            )
            return -1

    @staticmethod
    def find_files_in_folder(
        ext: str, folderpath: str, recursive: bool = True
    ) -> List[Path]:
        """Given a folder path, recursively searches that folder and all
        its subfolders for the given file extension. Returns a list of
        file Path objects for all matches.

        Args:
            ext (str): File extension of the filetype you wish to match.
                e.g. '.wav' or 'wav' will work.
            folderpath (str): Path to base directory to search for files.

        Returns:
            List[Path]: A list of Path objects of matching file paths.
        """

        # in case user did not provide '.' in `ext`
        if not ext.startswith("."):
            ext = "." + ext

        # find all files matching given extension
        if recursive:
            filetype = "**/*" + ext  # e.g. "**/*.wav"
        else:
            filetype = "*" + ext
        filelist = list(Path(folderpath).glob(filetype))

        return filelist

    @staticmethod
    def copyfiles(filepath_list: List[Path], dest_folder: str) -> List[Path]:
        """Given a list of filepaths, copies all into destination
        folder. Returns list of new filepaths at the destination folder.

        Args:
            filelist (List[Path]): List of file paths to be copied
            dest_folder (str): Destination folder to copy to

        Returns:
            List[Path]: List of file paths at destination path
        """
        dest_path = Path(dest_folder)
        dest_path.mkdir(
            parents=True, exist_ok=True
        )  # Create the destination folder if it doesn't exist

        newfilelist = []
        for filepath in tqdm(filepath_list):
            source_path = Path(filepath)
            dest_file = dest_path / source_path.name

            if not os.path.exists(dest_file):
                shutil.copy2(filepath, dest_file)
                logger.debug("copied %s", dest_file)
            else:
                logger.warning("%s exists, halted copy.", dest_file)
            newfilelist.append(dest_file)

        return newfilelist

    # ! - TODO - pull out into pathing.py
    @staticmethod
    def replace_path(
        old_path: Union[Path, str],
        new_subfolder: str = None,
        new_filename: str = None,
        new_ext: str = None,
    ) -> Path:
        """Given the above arguments, replaces the filename, extension
        and topmost subfolder in the path.

        Args:
            old_path (Path | str): Old file path.
            new_subfolder (str, optional): Topmost parent folder to
                rename to. Defaults to None. If None, does not rename.
            new_filename (str, optional): Filename to rename to.
                Defaults to None. If None, does not rename filename.
            new_fileext (str, optional): File extension to rename to.
                Defaults to None. If None, does not change extension.

        Returns:
            Path: Modified file path

        Example:
            >>> replace_path(
                'path/to/folder/abc.txt',
                'newfolder',
                'def',
                '.xml'
            )

            Path('path/to/newfolder/def.xml')

        """
        old_path = Path(old_path)
        old_filename = old_path.stem
        old_ext = old_path.suffix
        newpath = None

        # replace subfolder?
        if new_subfolder is not None:
            basepath = old_path.parent.parent
            newpath = basepath.joinpath(new_subfolder)
            newpath.mkdir(parents=True, exist_ok=True)
        else:
            newpath = old_path.parent

        if new_ext is not None and not str(new_ext).startswith("."):
            new_ext = "." + new_ext

        # replace filename?
        if new_filename is not None:
            # replace file ext?
            if new_ext is not None:
                newpath = newpath.joinpath(new_filename + new_ext)
            else:
                newpath = newpath.joinpath(new_filename + old_ext)
        else:
            if new_ext is not None:
                newpath = newpath.joinpath(old_filename + new_ext)
            else:
                newpath = newpath.joinpath(old_filename + old_ext)

        return newpath

    @staticmethod
    def delete_file(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            logger.error("Unable to delete %s : %s", filepath, e)

    def update_rttm_fileids(self, ali_far_path: Union[str, Path]):
        """By default, Ali Far rttms named as X####_X####.rttm whereas
        wav files named as X####_X####_A####.wav. After renaming using
        match_rename_far_rttms_to_audio(), use this to update RTTM file
        contents to reflect the correct file_id.
        """
        for split in ["train", "val", "test"]:
            ali_far_rttm_paths = (
                Path(ali_far_path).joinpath(split).joinpath("rttm").glob("*.rttm")
            )
            for rttm_path in ali_far_rttm_paths:
                self.update_rttm_fileid(rttm_path)

    def update_rttm_fileid(self, rttm_path):
        """After some renaming of audio or RTTM files, the RTTM contents
        itself still refers to the old file_id and will not match. Once
        an RTTM file is named correctly to match its corresponding
        audio, run this to update its contents to correct the file_id's
        in all of its speech segments.
        """
        rttm_path = Path(rttm_path)
        speech_segments = sseg.read_rttm(rttm_path)
        sseg.write_rttm(
            speech_segments,
            rttm_path,
            file_id=Path(rttm_path).stem,
        )


class AmiPrepDataFolders(PrepDataFolders):
    """
    Extends PrepDataFolders base class for AMI dataset-specific
    functionality.

    """

    def __init__(self, data_splits: Dict):
        super().__init__()

        self._TRAIN_SAMPLE_IDS = data_splits["train_ids"]
        self._VALIDATION_SAMPLE_IDS = data_splits["val_ids"]
        self._TEST_SAMPLE_IDS = data_splits["test_ids"]

        self._SAMPLE_IDS = {
            "train": self._TRAIN_SAMPLE_IDS,
            "val": self._VALIDATION_SAMPLE_IDS,
            "test": self._TEST_SAMPLE_IDS,
        }

    def train_val_test_split(self, filelist: List[Path]) -> List[List[Path]]:
        """Given a list of filepaths, refer against predefined lists of
        train/val/test splits and split the list accordingly into
        separate train, val, and test file lists.

        Args:
            filelist (List[Path]): a list of file Paths

        Returns:
            List[List[Path]]: _description_
        """
        train_filepaths = []
        val_filepaths = []
        test_filepaths = []

        for filepath in filelist:
            for file_id in self._TRAIN_SAMPLE_IDS:
                match = re.search(file_id, str(filepath))
                if match:
                    train_filepaths.append(filepath)

            for file_id in self._VALIDATION_SAMPLE_IDS:
                match = re.search(file_id, str(filepath))
                if match:
                    val_filepaths.append(filepath)

            for file_id in self._TEST_SAMPLE_IDS:
                match = re.search(file_id, str(filepath))
                if match:
                    test_filepaths.append(filepath)

        return train_filepaths, val_filepaths, test_filepaths

    @staticmethod
    def rename_xmlfiles(filelist: List[Path]) -> List[Path]:
        """Given a list of AMI xml filepaths, rename the ending suffix
        to match the AMI near field wave file suffixes.

        Currently they are:
            EN2001a.A.segments.xml
            EN2001a.Headset-0.wav

        We will rename to:
            EN2001a.Headset-0.xml * match the XML to the wave file name
            EN2001a.Headset-0.wav


        Args:
            filelist (List[Path]): list of XML file paths

        Returns:
            List[str]: List of renamed XML file paths.
        """
        newfilelist = []
        for filepath in filelist:
            new_filepath = str(filepath)
            if str(filepath).endswith(".A.segments.xml"):
                new_filepath = str(filepath).replace(
                    ".A.segments.xml", ".Headset-0.xml"
                )
            elif str(filepath).endswith(".B.segments.xml"):
                new_filepath = str(filepath).replace(
                    ".B.segments.xml", ".Headset-1.xml"
                )
            elif str(filepath).endswith(".C.segments.xml"):
                new_filepath = str(filepath).replace(
                    ".C.segments.xml", ".Headset-2.xml"
                )
            elif str(filepath).endswith(".D.segments.xml"):
                new_filepath = str(filepath).replace(
                    ".D.segments.xml", ".Headset-3.xml"
                )
            elif str(filepath).endswith(".E.segments.xml"):
                new_filepath = str(filepath).replace(
                    ".E.segments.xml", ".Headset-4.xml"
                )
            elif str(filepath).endswith(".F.segments.xml"):
                new_filepath = str(filepath).replace(
                    ".F.segments.xml", ".Headset-5.xml"
                )
            elif str(filepath).endswith(".G.segments.xml"):
                new_filepath = str(filepath).replace(
                    ".G.segments.xml", ".Headset-6.xml"
                )
            elif str(filepath).endswith(".H.segments.xml"):
                new_filepath = str(filepath).replace(
                    ".H.segments.xml", ".Headset-7.xml"
                )
            elif str(filepath).endswith(".I.segments.xml"):
                new_filepath = str(filepath).replace(
                    ".I.segments.xml", ".Headset-8.xml"
                )

            newfilelist.append(new_filepath)

            shutil.move(str(filepath), new_filepath)

        return newfilelist


class AliPrepDataFolders(PrepDataFolders):
    """
    Extends PrepDataFolders base class for Ali dataset-specific
    functionality.

    """

    def __init__(self):
        super().__init__()
        return

    # ! - TODO catch error for name not found in train/val/test.
    # or if any are missing.
    def match_rename_far_textgrid_to_audio(self, ali_far_path: Union[Path, str]):
        """By default, Ali Far textgrids named as X####_X####.textgrid
        whereas wav files named as X####_X####_A####.wav. This renames
        the textgrids to match the wave files.
        """
        basepath = Path(ali_far_path)

        for split in ["train", "val", "test"]:
            print(basepath.joinpath(split))

            audio_paths = basepath.joinpath(split).joinpath("audio").glob("*.wav")
            tgrid_paths = (
                basepath.joinpath(split).joinpath("textgrid").glob("*.TextGrid")
            )
            audio_paths = sorted(audio_paths)
            tgrid_paths = sorted(tgrid_paths)

            for audio_path in audio_paths:
                for tgrid_path in tgrid_paths:
                    result = re.search(
                        str(tgrid_path.stem),
                        str(audio_path.stem),
                    )
                    if result:
                        os.rename(
                            tgrid_path,
                            tgrid_path.parent.joinpath(
                                audio_path.stem + tgrid_path.suffix
                            ),
                        )

    def match_rename_far_rttms_to_audio(self, ali_far_path: Union[Path, str]):
        """By default, Ali Far rttms named as X####_X####.rttm whereas
        wav files named as X####_X####_A####.wav. This renames the rttms
        to match the wave files.
        """
        basepath = Path(ali_far_path)

        results = {}
        for split in ["train", "val", "test"]:
            # print(basepath.joinpath(split))

            audio_paths = basepath.joinpath(split).joinpath("audio").glob("*.wav")
            rttm_paths = basepath.joinpath(split).joinpath("rttm").glob("*.rttm")
            audio_paths = sorted(audio_paths)
            rttm_paths = sorted(rttm_paths)

            new_rttm_paths = []
            for audio_path in audio_paths:
                for rttm_path in rttm_paths:
                    result = re.search(
                        str(rttm_path.stem),
                        str(audio_path.stem),
                    )
                    if result:
                        new_rttm_path = rttm_path.parent.joinpath(
                            audio_path.stem + rttm_path.suffix
                        )
                        os.rename(
                            rttm_path,
                            new_rttm_path,
                        )
                        new_rttm_paths.append(new_rttm_path)

            results[split] = new_rttm_paths

        return results

    def split_far_multichannel(self, ali_far_path: Union[str, Path]):
        """Specifically for Ali Far, all the audio files are actually
        8-channel wave files. This splits them into 8 mono files named
        <filename>1.wav to <filename>8.wav. Also duplicates the RTTM
        files to match.
        """
        ali_far_path = Path(ali_far_path)

        for split in [
            "test",
            "train",
            "val",
        ]:
            audio_paths = ali_far_path.joinpath(split).joinpath("audio").glob("*.wav")

            # for each audio/rttm pair
            for audio_path in audio_paths:
                # check how many channels
                info = sf.info(str(audio_path))
                num_channels = int(info.channels)

                if num_channels > 1:
                    signal, sr = self.read_audio(audio_path)

                    rttm_path = self.toggle_audio_rttm_path(audio_path)

                    # split n ways:
                    for channel in range(num_channels):
                        # make new audio file
                        signal_ch = signal[:, channel]
                        audio_ch_path = audio_path.parent.joinpath(
                            str(audio_path.stem)
                            + "_"
                            + str(channel + 1)
                            + str(audio_path.suffix)
                        )
                        status = self.save_audio(audio_ch_path, signal_ch, sr)
                        # if error saving audio (it should have already thrown error)
                        # but just additionally break the loop here.
                        if status == -1:
                            break

                        # make new RTTM
                        rttm_ch_path = rttm_path.parent.joinpath(
                            str(audio_path.stem) + "_" + str(channel + 1) + ".rttm"
                        )

                        status = self.duplicate_rttm_file(rttm_path, rttm_ch_path)
                        if status == -1:
                            break

                        # success
                        logger.info(
                            "Saved channel %s as %s", channel + 1, audio_ch_path
                        )

                    del signal, sr
                    self.delete_file(rttm_path)
                    self.delete_file(audio_path)
