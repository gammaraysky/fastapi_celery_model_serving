"""For reading in a data folder comprising of `train`, `val`, `test`
subfolders, each containing `audio` and `rttm` folders which contain
.wav and .rttm files respectively.

    +-- train/
    |   +-- audio/*.wav ...
    |   +-- rttm/*.rttm ...
    +-- val/
    |   +-- audio/
    |   +-- rttm/
    +-- test/
    |   +-- audio/
    |   +-- rttm/

"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.vad.data_prep import speech_segments as sseg

logger = logging.getLogger(__name__)


class DataLoadFolders:
    """Reads in a an audio Dataset of matching .wav and .rttm pairs,
    already split into train, val, and test sets.

    Filenames for .wav and .rttm pairs must match.

    Dataset can be exported to a dict using self.to_dict()
    """

    def __init__(self, data_path: Union[str, Path]) -> None:
        """Initialize DataLoadFolders object. Reads in 'train', 'val'
        and 'test' folders in the given `data_path` and for each one,
        read in .wav files in 'audio' subfolder, rttm files in 'rttm
        subfolder. Only matching pairs of audio and rttm files where
        the names match are read in.

        Args:
            data_path (str | Path): Base path to data folders. Data
                must be structured in the following format:

                <data_path>/train/audio/ (contains *.wav files)
                <data_path>/train/rttm/ (contains *.rttm files)
                <data_path>/val/audio/ (contains *.wav files)
                <data_path>/val/rttm/ (contains *.rttm files)
                <data_path>/test/audio/ (contains *.wav files)
                <data_path>/test/rttm/ (contains *.rttm files)

        """

        self.data_path = Path(data_path)
        self.train, dangling_train = self.load_files_in_split(
            "train", return_dangling=True
        )
        self.val, dangling_val = self.load_files_in_split("val", return_dangling=True)
        self.test, dangling_test = self.load_files_in_split(
            "test", return_dangling=True
        )

        self.dangling_files = dangling_train + dangling_val + dangling_test

    def get_dangling_files(self) -> List:
        """Returns a list of dangling files where the audio is missing
        its corresponding rttm file, or vice versa.

        Returns:
            List: list of filepaths where the corresponding counterpart
                is missing.
        """
        return self.dangling_files

    def save_dangling_files_to_json(self, json_path) -> None:
        """Saves a list of dangling files where the audio is missing
        its corresponding rttm file, or vice versa, as a JSON file.
        """

        with open(json_path, "a") as file:
            json.dump(self.dangling_files, file, indent=1)

    def to_dict(self) -> Dict:
        """Returns validated dataset as a Dict

        Returns:
            Dict: Validated dataset, in the following format:
            {
                'train' : {
                    <fileid1> : {
                        'audio_path' (Path) : path to wav file
                        'rttm_path' (Path) : path to matching rttm file
                        'segments' (List of Tuples) : speech segments
                            already read in from rttm file
                    <fileid2> : { }
                    <fileid3> : { }
                    }
                }
                'val' : { same as 'train'}
                test : { same as 'train'}
            }

            where fileids refer to the file stems for each audio file.
        """
        dataset_dict = {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }

        return dataset_dict

    def load_files_in_split(
        self, split: str, return_dangling: bool = True
    ) -> Union[Dict, List]:
        """Reads in matching .wav and .rttm files in `<split>` folder
        and returns them as a dict. Unmatched/dangling files which are
        missing their counterpart .wav or .rttm file are not read in.

        Args:
            split (str): 'train', 'val', or 'test'

        Returns:
            Dict: a Dict of matching audio and rttm pairs as follows:
                {
                  '<fileid1>' : {
                    'audio_path' : /path/to/audio/<fileid1>.wav
                    'rttm_path : /path/to/rttm/<fileid1>.rttm
                    'segments' : List of Tuples of speech segments read
                        in from rttm file
                  }
                  '<fileid2> : { }
                  '<fileid3> : { }
                  ...
                }
        """
        audio_paths = list(
            self.data_path.joinpath(split).joinpath("audio").glob("*.wav")
        )
        rttm_paths = list(
            self.data_path.joinpath(split).joinpath("rttm").glob("*.rttm")
        )

        file_ids, dangling_paths = self._get_matching_file_pairs(
            audio_paths, rttm_paths, return_dangling=True
        )

        loaded_items = {}
        for file_id in file_ids:
            audio_path, rttm_path, segments = self._load_item(split, file_id)
            file_id = str(file_id)

            loaded_items[file_id] = {
                "audio_path": audio_path,
                "rttm_path": rttm_path,
                "segments": segments,
            }

        if return_dangling:
            return loaded_items, dangling_paths
        else:
            return loaded_items

    def _load_item(self, split: str, file_id: str) -> Tuple:
        """Given a fileid (filename stem), generates audio path, rttm
        path, and reads in the rttm file as a list of segment tuples.

        Args:
            split (str): 'train', 'val', or 'test'
            file_id (str): string denoting the fileid aka filename stem
                (without the extension)

        Returns:
            Tuple[Path, Path, List[[Tuple[float]]]]: returns an audio
                path, an rttm path, and a list of segment tuples of
                speech start and end times in seconds.
        """

        audio_path = str(
            self.data_path.joinpath(split).joinpath("audio").joinpath(file_id + ".wav")
        )
        rttm_path = str(
            self.data_path.joinpath(split).joinpath("rttm").joinpath(file_id + ".rttm")
        )
        segments = sseg.read_rttm(rttm_path)
        segments = sseg.validate_segments(segments)

        return (audio_path, rttm_path, segments)

    def _get_matching_file_pairs(
        self,
        audio_paths: List[Path],
        rttm_paths: List[Path],
        return_dangling: bool = True,
    ) -> Union[List[Path], Tuple[List[Path], List[Path]]]:
        """Given a list of audio_paths and a list of rttm_paths, finds
        files that have matching pairs in both lists. e.g.
            audio_paths : [... 'path/audio/file01.wav' ... ]
            rttm_paths : [... 'path/rttm/file01.rttm' ... ]

        Args:
            audio_paths (List[Path]): list of filepaths to .wav files
            rttm_paths (List[Path]): list of filepaths to .rttm files
            return_dangling (bool, optional): If True, returns a list
                of filepaths where the counterpart file was not found.
                Defaults to True.

        Returns:
            Either returns:
            -   List[Path] - list of common fileids (file stems) that
                    match between audio and rttm lists

            or:
            - Tuple[List[Path], List[Path]] - tuple of the above list
                    and another list:
                        List[Path] of dangling filepaths where unmatched

        """
        # check mismatch files using sets and difference
        audio_path_stems = [item.stem for item in audio_paths]
        rttm_path_stems = [item.stem for item in rttm_paths]

        audio_path_stems = set(audio_path_stems)
        rttm_path_stems = set(rttm_path_stems)

        # audio exists, rttm does not
        missing_rttm_stems = list(audio_path_stems - rttm_path_stems)
        missing_rttm_paths = [
            path
            for path in rttm_paths
            if any(name in path for name in missing_rttm_stems)
        ]

        # rttm exists, audio does not
        missing_audio_stems = list(rttm_path_stems - audio_path_stems)
        missing_audio_paths = [
            path
            for path in audio_paths
            if any(name in path for name in missing_audio_stems)
        ]

        # present in both
        common_stems = list(audio_path_stems.intersection(rttm_path_stems))

        if len(missing_audio_stems) > 0 or len(missing_rttm_stems) > 0:
            logger.warning("mismatch between audio and rttm files detected.")
            if len(missing_audio_stems) > 0:
                logger.warning("missing audio files at %s:", audio_paths[0].parent)
                logger.warning(missing_audio_paths)
            if len(missing_rttm_stems) > 0:
                logger.warning("missing rttm files at %s:", rttm_paths[0].parent)
                logger.warning(missing_rttm_paths)

        if return_dangling:
            return common_stems, missing_audio_paths + missing_rttm_paths
        else:
            return common_stems
