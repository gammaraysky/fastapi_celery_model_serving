"""speech_distributions.py

For calculation of distribution of speech and nonspeech proportions
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import logging

import numpy as np
import soundfile as sf

from . import speech_segments as sseg


logger = logging.getLogger(__name__)


class SpeechDistributions:
    def get_speech_nonspeech_durations_wavefile(
        self, audio_path: Union[Path, str], speech_segments_secs: Tuple[float]
    ) -> Tuple[float]:
        """Calculates speech and nonspeech durations per audio file

        Args:
            audio_path (Union[Path,str]): Path to audio file
            speech_segments_secs (List[Tuple[float]]): Speech segments
                in a list of tuples of start and end times for each
                segment, in seconds.

        Returns:
            Tuple[float]: (speech_duration, nonspeech_duration) in
                seconds.
        """
        # get audio duration
        duration_secs = sf.info(audio_path).duration
        # invert segments to get nonspeech
        speech_segments_secs = sseg.merge_overlap_segments([speech_segments_secs])
        nonspeech_segments_secs = sseg.invert_segments(
            speech_segments_secs, duration_secs
        )
        # collate segment durations - speech
        speech_duration = sseg.total_segment_duration(speech_segments_secs)
        # collate segment durations - nonspeech
        nonspeech_duration = sseg.total_segment_duration(nonspeech_segments_secs)

        if speech_duration is None or nonspeech_duration is None:
            logger.error("%s, %s, %s", audio_path, speech_duration, nonspeech_duration)

        return speech_duration, nonspeech_duration

    def get_speech_nonspeech_durations_in_datasplit(
        self, datasplit: Dict
    ) -> Tuple[float]:
        """Calculates speech and nonspeech durations over a dataset
        split e.g. AMI far train split.

        Args:
            datasplit (Dict): a Dict with fileids as keys, containing
                subkeys 'audio_path', 'rttm_path' and 'segments'.
                (Use DataLoadFolders.to_dict() to generate this.)

        Returns:
            Tuple[float]: (speech_duration, nonspeech_duration) in
                seconds.
        """
        speech_durations = []
        nonspeech_durations = []
        for fileid in datasplit.keys():
            spc, nonspc = self.get_speech_nonspeech_durations_wavefile(
                datasplit[fileid]["audio_path"], datasplit[fileid]["segments"]
            )
            speech_durations.append(spc)
            nonspeech_durations.append(nonspc)

        return (
            np.sum(speech_durations),
            np.sum(nonspeech_durations),
        )

    def get_speech_nonspeech_durations_across_splits(self, dataset: Dict) -> Dict:
        """Calculates speech and nonspeech durations on each split of a
        dataset, for comparative analysis.

        Args:
            dataset (Dict): a Dict with train/val/test split as keys,
                fileids as subkeys, and 'audio_path', 'rttm_path' and \
                'segments' as next subkeys.
                (Use DataLoadFolders.to_dict() to generate this.)

        Returns:
            Dict: a dict with 'train'/'val'/'test' as keys, and
                'speech'/'nonspeech' as subkeys, and durations in
                seconds (float) for each value.
        """
        durations = {}
        for split in ["train", "val", "test"]:
            spc, nonspc = self.get_speech_nonspeech_durations_in_datasplit(
                dataset[split]
            )

            total = spc + nonspc

            durations[split] = {
                "speech": round(spc, 2),
                "nonspeech": round(nonspc, 2),
                "spc_prop": round(spc / total, 4),
                "nonspc_prop": round(nonspc / total, 4),
                "total": round(total, 2),
            }

        return durations
