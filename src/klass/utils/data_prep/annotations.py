import logging
import os
from os.path import join
from typing import List, Tuple

import hydra
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from textgrid import TextGrid


class Annotations:
    """
    A class that generates annotations from AMI csv and Alimeeting
    textgrid files.

    This class handles the mapping of audio file paths to start and end
    speech segments for both near-field and far-field recordings, based
    on the paths provided in the annotations_path yaml file.

    Attributes:
        audio_data (List[str]): Contains audio_path and segments fields.
        configuration (dict): Configuration data loaded from the
            provided yaml file.
        ami_far_csv_path (str): File path to the AMI far-field
            annotations.
        ami_near_csv_path (str): File path to the AMI near-field
            annotations.
        ami_audio_far_root (str): Root directory for far-field audio
            files.
        ami_audio_near_root (str): Root directory for near-field audio
            files.
    """

    def __init__(self, data_annotations_path: str):
        """
        Initializes Annotations from annotation data.

        Args:
            data_annotations_path (str): Path to the yaml
                annotation_paths file.

        Examples:
            if config file : /dir/config.yaml
            ensure extension is .yaml and not yml
            ensure that config_dir=/dir and config_name="config"
        """

        with hydra.initialize_config_dir(
            config_dir=data_annotations_path, job_name="annotations"
        ):
            annotation_paths = hydra.compose(config_name="annotation_paths")["paths"]
            logging.info("Hydra config initialized: {}".format(annotation_paths))

        self.paths = annotation_paths

        self.ami_far_csv_path = os.path.join(
            annotation_paths["ami_far_root_folder"],
            annotation_paths["ami_far_annot_filename"],
        )

        self.ami_annot_near_csv_path = os.path.join(
            annotation_paths["ami_near_root_folder"],
            annotation_paths["ami_near_annot_filename"],
        )

        self.ami_audiofile_far_root = os.path.join(
            annotation_paths["ami_far_root_folder"],
            annotation_paths["ami_audiofile_head"],
            # annotation_paths["far_field_file_head"],
        )

        self.ami_audiofile_near_root_folder = annotation_paths["ami_near_root_folder"]

    def ami_annotations_loader(self):
        """
        Loads AMI configs into a dictionary.

        Returns:
            dict: A dictionary containing grouped data for each near and
                far recording. Each entry is a dictionary containing
                audio_path and start, end segments.

        Examples:
            annotation files:
                Near-field dataframe:
                    id,start,end,split
                    IS1009d.Headset-1,41.117,41.68,test

                Far-field dataframe:
                    id,start,end,start_min,end_min,split
                    EN2001a,3.339999914,3.880000114,00:03.3,00:03.9,train

        """
        data = {}
        for label in ["near", "far"]:
            far = True if label == "far" else False
            annote_to_df = self._ami_annot_w_file_paths(far=far)

            annote_to_df_group = annote_to_df.groupby("split")
            for train_val_test, respective_df in annote_to_df_group:
                train_val_test = f"ami_{train_val_test}_{label}"
                data[train_val_test] = {}
                id_grp = respective_df.groupby("id")
                for id_grp_name, id_grp_df in id_grp:
                    start_end_list = list(
                        zip(id_grp_df["start"].values, id_grp_df["end"].values)
                    )
                    start_end_list_round = list(
                        map(
                            lambda start_vs_end_time: (
                                round(start_vs_end_time[0], 4),
                                round(start_vs_end_time[1], 4),
                            ),
                            start_end_list,
                        )
                    )

                    audio_path = id_grp_df["file_paths"].unique()[0]

                    duration_secs = librosa.get_duration(path=audio_path)
                    sr = librosa.get_samplerate(path=audio_path)
                    duration_samples = int(duration_secs * sr)

                    # for each file's list of segments, add class labels
                    # as a 'signal' at 100 labels/frames per second.
                    speech_labels = self.convert_segments_to_signal(
                        duration_samples,
                        100,
                        start_end_list_round,
                    )

                    data[train_val_test][id_grp_name] = {
                        "audio_path": audio_path,
                        "segments": start_end_list_round,
                        "speech_labels": speech_labels,
                    }

        return data

    def _each_far_field_array_and_their_audio_path(
        self, far_field_annot_df, array_folder_name
    ):
        """
        Original Annotations file for far-field was for Array1-01.
        Now we pair the Annotation to all.

        Args:
            far_field_annot_df (pd.DataFrame): Dataframe of annotations
                for far field.
            array_folder_name (str): Name of the far-field array folder.

        Returns:
            pd.DataFrame: Dataframe with file paths and ids added.
        """

        far_field_annot_df_clone = far_field_annot_df.copy(deep=True)
        file_path_column_name = "file_paths"
        far_field_annot_df_clone[file_path_column_name] = far_field_annot_df_clone[
            "id"
        ].apply(
            lambda filename: join(
                join(self.ami_audiofile_far_root, array_folder_name),
                filename,
                "audio",
                f"{filename}.{array_folder_name}.wav",
            )
        )
        far_field_annot_df_clone["id"] = far_field_annot_df_clone[
            file_path_column_name
        ].apply(lambda audio_path: os.path.basename(audio_path))
        return far_field_annot_df_clone

    def _get_far_field_path(self, far_field_annot_df):
        """
        Generates the file path for audios of far field array-01.

        Args:
            far_field_annot_df (pd.DataFrame): Dataframe containing
                information about the far-field recordings.

        Returns:
            pd.DataFrame: Dataframe with file paths and ids added.
        """

        all_far_field_array_folder_name = [
            array_folder_name
            for array_folder_name in os.listdir(self.ami_audiofile_far_root)
            if "Array" in array_folder_name
        ]

        list_of_df_of_far_field_info = []

        for indiv_far_field_array_folder_name in all_far_field_array_folder_name:
            list_of_df_of_far_field_info.append(
                self._each_far_field_array_and_their_audio_path(
                    far_field_annot_df, indiv_far_field_array_folder_name
                )
            )

        far_field_account_for_all_arrays = pd.concat(
            list_of_df_of_far_field_info, ignore_index=True, sort=False
        )

        return far_field_account_for_all_arrays

    def _get_near_field_path(self, near_field_annot_df):
        """
        Transforms the dataframe for near-field recordings by adding
            file paths and ids.

        Args:
            near_field_annot_df (pd.DataFrame): Dataframe containing
                information about the near-field recordings.

        Returns:
            pd.DataFrame: Transformed dataframe with file paths and ids
                added.
        """

        near_field_annot_df["file_paths"] = near_field_annot_df["id"].apply(
            lambda element: os.path.join(
                self.ami_audiofile_near_root_folder,
                element[: element.index(".")],
                "audio",
                f"{element}.wav",
            )
        )

        near_field_annot_df["id"] = near_field_annot_df["file_paths"].apply(
            lambda file_path: os.path.basename(file_path)
        )

        return near_field_annot_df

    def _ami_annot_w_file_paths(self, far=True) -> pd.DataFrame():
        """
        Generates and adds audio file paths to annotations dataframe

        Args:
            far (bool, optional): Flag indicating whether to generate
                far-field audio file paths. Default is True.

        Returns:
            pd.DataFrame: Dataframe containing the added audio file
                paths.
        """

        if far:
            ami_far_to_df = pd.read_csv(self.ami_far_csv_path)
            df_far_field = self._get_far_field_path(ami_far_to_df)
            return df_far_field
        else:
            ami_near_to_df = pd.read_csv(self.ami_annot_near_csv_path)
            df_near_field = self._get_near_field_path(ami_near_to_df)
            return df_near_field

    def ali_annotations_loader(self):
        """
        Loads Alimeeting configs into a dictionary.

        Returns:
            dict: A dictionary containing grouped data for each
                recording in the dataset.
        """

        data = {}

        keys = ["train", "validation", "test"]
        far = [True, False]
        for condition in far:
            for train_val_test in keys:
                (
                    audio_dir,
                    textgrid_dir,
                    new_train_val_test_naming,
                ) = self._get_ali_paths(far=condition, train_val_test=train_val_test)

                name_textgrid_audio = self._get_all_audio_n_grid_path(
                    audio_dir, textgrid_dir
                )
                # meta_data_of_audiopath_textgridpath_segments = {
                #     path_n_segments[0]: {
                #         "audio_path": path_n_segments[1],
                #         "textgrid_path": path_n_segments[2],
                #         "segments": self._remove_overlap(path_n_segments[2]),
                #         "speech_labels":
                #     }
                #     for path_n_segments in name_textgrid_audio
                # }
                meta_data_of_audiopath_textgridpath_segments = {}
                for path_n_segments in name_textgrid_audio:
                    duration_secs = librosa.get_duration(path=path_n_segments[1])
                    sr = librosa.get_samplerate(path=path_n_segments[1])
                    duration_samples = int(duration_secs * sr)

                    # for each file's list of segments, add class labels
                    # as a 'signal' at 100 labels/frames per second.
                    segments = self._remove_overlap(path_n_segments[2])

                    speech_labels = self.convert_segments_to_signal(
                        duration_samples,
                        100,
                        segments,
                    )

                    meta_data_of_audiopath_textgridpath_segments[path_n_segments[0]] = {
                        "audio_path": path_n_segments[1],
                        "textgrid_path": path_n_segments[2],
                        "segments": segments,
                        "speech_labels": speech_labels,
                    }

                data[
                    new_train_val_test_naming
                ] = meta_data_of_audiopath_textgridpath_segments

        return data

    def _get_ali_paths(self, far: bool = False, train_val_test: str = "train"):
        """
        Gets the folder paths of alimeet audio files and textgrids.

        Args:
            far (bool, optional): Flag indicating whether to get paths
                for far-field or near-field. Default is False.
            train_val_test (str, optional): Indicates whether the paths
                are for the "train", "validate", or "test" set. Default
                is "train".

        Returns:
            tuple: Tuple containing the paths to the audio and textgrid
                directories, and the updated sample name.
        Example:
            audio_dir, textgrid_dir, ali_far_near = self._get_ali_paths(
                far=False,Train)

            audio_dir : /polyaxon-v1-data/workspaces/data/20230414_1516813_Ali/Alimeeting_Dataset/Train_Ali_near/audio_dir
            textgrid_dir : /polyaxon-v1-data/workspaces/data/20230414_1516813_Ali/Alimeeting_Dataset/Train_Ali_near/textgrid_dir
            ali_far_near : Train
        """
        head = self.paths["ali_dataset_root"]
        train_val_test = train_val_test.lower()
        if far:
            ali_far_near = f'ali_{train_val_test}_{"far"}'
        else:
            ali_far_near = f'ali_{train_val_test}_{"near"}'

        if ali_far_near not in self.paths:
            raise ValueError(
                f"{ali_far_near} not in config file because {train_val_test} is not"
                " valid "
            )

        selected_path = self.paths[ali_far_near]

        audio_dir = join(head, selected_path, self.paths["ali_audio_dir"])
        textgrid_dir = join(head, selected_path, self.paths["ali_tg_dir"])

        return audio_dir, textgrid_dir, ali_far_near

    def _create_list_of_paths_from_dir(self, _dir):
        """
        Creates a list of all file paths in the given directory.

        Args:
            _dir (str): The directory path.

        Returns:
            list: List of file paths.
        """
        filenames = os.listdir(_dir)
        paths = [join(_dir, _) for _ in filenames]
        filtered = [path for path in paths if os.path.isfile(path)]
        return filtered

    def _get_all_audio_n_grid_path(self, audio_dir, textgrid_dir):
        """
        Gets all audio and grid paths after getting ali_paths.

        Args:
            audio_dir (str): Directory path of audio files.
            textgrid_dir (str): Directory path of grid files.

        Returns:
            list: List of all audio and grid paths.
        """
        all_wav_path = self._create_list_of_paths_from_dir(audio_dir)
        all_textgrid_path = self._create_list_of_paths_from_dir(textgrid_dir)
        all_wav_path.sort()
        all_textgrid_path.sort()
        all_wav_name = [os.path.basename(path) for path in all_wav_path]
        all_tg_name = [os.path.basename(path) for path in all_textgrid_path]

        map_textgrid_aud = [
            [
                all_wav_name[index].split(".")[0],
                all_wav_path[index],
                all_textgrid_path[index],
            ]
            for index in range(len(all_tg_name))
            if all_tg_name[index].split(".")[0] in all_wav_name[index]
        ]

        return map_textgrid_aud

    def _get_speech_segments_from_textgrid(self, textgrid_filepath):
        """
        Extracts speech segments from a TextGrid file.

        Args:
            textgrid_filepath (str): Path to the TextGrid file.

        Returns:
            list: List of tuples representing start and end
                  times of each speech segment.
        """
        # parse the textgrid file
        textgrid = TextGrid()
        if os.path.isfile(textgrid_filepath):
            textgrid.read(textgrid_filepath)

        speech_segments = []

        for tier in range(len(textgrid.tiers)):
            try:
                for interval in textgrid.tiers[tier]:
                    start = interval.minTime
                    end = interval.maxTime
                    speech_segments.append((start, end))
            except Exception as e:
                logging.error(e)
                logging.error("Could not process: {}".format(textgrid_filepath))
                break
        return speech_segments

    def _remove_overlap(self, textgrid_file):
        """
        Removes overlapping segments from a textgrid file.

        Args:
            textgrid_file (str): Path to the TextGrid file.

        Returns:
            list: List of tuples representing non-overlapping start and
                end times of each speech segment.

        Examples:

            Input = [(5.10,6.5),(6,8),(9,10)]
            result = [(5.10,8),(9,10)]
            because 6 is within 5.10 to 6.5
        """
        segments = self._get_speech_segments_from_textgrid(textgrid_file)
        segments.sort(
            key=lambda speech: speech[0]
        )  # Sort the intervals based on start times

        result = []  # Store the processed intervals

        for interval in segments:
            if not result or interval[0] > result[-1][1]:  # No overlap
                result.append(interval)
            else:  # Overlap exists
                last_interval = result[-1]
                if interval[1] > last_interval[1]:
                    result[-1] = (last_interval[0], interval[1])
        return result

    def invert_segments(
        self, segments_sec: list, duration_samples: int, sr: int
    ) -> list:
        """Given a list of speech segments, and duration of signal,
        generates the inverse, i.e. nonspeech segments.

        Used for Alimeeting dataset as its default segment annotations
        only covers speech segments.

        Args:
            segments_sec (list of tuples): List of start and end times
                (in seconds) for samples containing speech.
            duration (int): Duration of signal, in samples
            sr (int): Sampling rate. Will be used to convert back
                duration into seconds.

        Returns:
            list: A list of tuples of start and end samples of nonspeech
                segments

        Examples:
            invert_segments(
                [(1.1, 2.2), (5.5, 6.6), (7.7, 8.8)], 220500, 22050)

            Output: [(0, 1.1), (2.2, 5.5), (6.6, 7.7), (8.8, 10.0)]

            ---

            signal, sr = librosa.load(
                ali['ali_train_far']['A0001_M0001_MS006']['audio_path'])
            invert_segments(
                ali['ali_train_far']['A0001_M0001_MS006']['segments],
                len(signal),
                sr)
        """

        duration_seconds = duration_samples / sr
        inverse_indices = []

        for i, (start, end) in enumerate(segments_sec):
            # if first speech segment and it does not start at sample 0
            # add in a nonspeech segment from sample 0 to current
            if i == 0 and start > 0:
                inverse_indices.append((0, start))

            # if last speech segment and it ends before last sample, add in
            # a nonspeech segment from current sample to the last sample
            if i == len(segments_sec) - 1 and end < duration_seconds:
                inverse_indices.append((end, duration_seconds))
                break

            # retrieve the next speech segment's start sample
            try:
                next_speech_start = segments_sec[i + 1][0]
            except Exception:
                next_speech_start = None
            # and create a nonspeech segment starting from the current end
            # to next segment's start.
            inverse_indices.append((end, next_speech_start))

        return inverse_indices

    def concat_signal_segments(
        self,
        segments_sec: list,
        signal_sr_tuple: Tuple[np.ndarray, int],
    ) -> np.array:
        """Given an audio signal and a list of segments, return a
        concatenated signal of just the selected segments.

        Args:
            segments_sec (list): List of tuples containing selection
                start and end indices.
            signal (np.array): Original audio signal
            sr (int): Audio sample rate in Hz e.g. 16000

        Returns:
            np.array: Concatenated audio signal of just selected
                segments.

        Examples:
            >>> segments_sec = [(0, 2.5), (10.0, 15.0)]
            >>> signal_sr_tuple = [([
                    55, 14, 72, 68, 99,
                    77, 40, 45, 76, 6,
                    23, 92, 66, 18, 61,
                    89
                ])]
            >>> anot = Annotations("data_annotations_path")
            >>> anot.concat_signal_segments(segments_sec, signal_sr_tuple)
            [55, 14, 23, 92, 66, 18, 61]
        """
        signal, sr = signal_sr_tuple

        concat_signal = []
        for start, end in segments_sec:
            start_sample = int(round(start * sr))
            end_sample = int(round(end * sr))
            concat_signal.extend(signal[start_sample:end_sample])

        return np.array(concat_signal)

    def get_speech_nonspeech_segments(
        self, signal_sr_tuple: Tuple[np.ndarray, int], segments: List
    ) -> tuple:
        """For a given audio file, extract all the speech segments and
        concatenate the signal to a new array. Also do this for the
        nonspeech segments.

        Args:
            data (dict): _description_

        Returns:
            tuple: 2 separate np.arrays of audio signal, first for
                speech and second for nonspeech

        Examples:
            >>> signal = [
                    55, 14, 72, 68, 99,
                    77, 40, 45, 76, 6,
                    23, 92, 66, 18, 61,
                ]
            >>> sample_rate = 2
            >>> segments = [(2.5, 4.0), (5.3, 7.1)]
            >>> anot = Annotations("data_annotations_path")
            >>> anot.get_speech_nonspeech_segments(
                    (
                        signal, sample_rate
                    ),
                    segments
                )
            [77, 40, 45, 92, 66, 18], [55, 14, 72, 68, 99, 76, 6, 23, 61]
        """

        signal, sr = signal_sr_tuple
        duration = len(signal)

        # data['segments_sec'] contains a list of speech segments_sec,
        # we will have to 'subtract' from the original signal,
        # to get the nonspeech segments_sec
        nonspeech_segments = self.invert_segments(segments, duration, sr)

        speech_array = self.concat_signal_segments(segments, (signal, sr))
        nonspeech_array = self.concat_signal_segments(nonspeech_segments, (signal, sr))

        return speech_array, nonspeech_array

    def convert_segments_seconds_to_samples(
        self,
        sr: int,
        segments: List,
    ) -> List:
        """Converts a list of segments denoted in seconds to a list of
        segments denoted in samples, based on sampling rate of given
        audio file.

        Args:
            sr (int): Audio sample rate in Hz e.g. 16000
            segments (List): List of segments denoted in seconds

        Returns:
            List: List of segments denoted in samples

        Examples:
            >>> sr = 2
            >>> segments = [(2.5, 10.0), (15.0, 36.0), (42.02, 60.77)]
            >>> anot = Annotations("data_annotations_path")
            >>> anot.convert_segments_seconds_to_sample(sr, segments)
            [(5, 20), (30, 72), (84, 122)]
        """

        segments_samples = []

        for starttime, endtime in segments:
            if isinstance(starttime, float) and isinstance(endtime, float):
                segments_samples.append(
                    (int(round(starttime * sr)), int(round(endtime * sr)))
                )

        return segments_samples

    def convert_segments_to_signal(
        self,
        duration_samples: int,
        sr: int,
        segments: List,
    ) -> np.array:
        """Converts a list of annotation segments into a signal array of
        0s and 1s, where 0 denotes absence and 1 denotes presence of a
        segment.

        Args:
            duration (int): Duration of signal, in samples
            sr (int): Audio sample rate in Hz e.g. 16000
            segments (List): List of segments denoted in seconds

        Returns:
            np.array: Annotated signal where 1 denotes segment is
                present.

        Examples:
            >>> duration_samples = 20
            >>> sample_rate = 2
            >>> merged_intervals = [(2.5, 4), (7.1, 9)]
            >>> anot = Annotations("data_annotations_path")
            >>> anot.convert_segments_to_signal(
                duration_samples,
                sample_rate,
                merged_intervals
            )
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
        """

        # convert the segments denoted in seconds to denoted in samples first
        segments_samples = self.convert_segments_seconds_to_samples(sr, segments)

        # Initialize the signal array with zeros
        annotation_signal = np.zeros(duration_samples)

        # Set the corresponding segments in the signal array to 1
        for segment_start, segment_end in segments_samples:
            annotation_signal[segment_start:segment_end] = 1

        return annotation_signal

    def plot_waveform_annotations(
        self,
        signal_sr_tuple: Tuple[np.ndarray, int],
        segments: List,
        start_sample: int = None,
        end_sample: int = None,
        label: str = None,
        ax: plt.axes = None,
    ) -> None:
        """Plots an audio waveform in the time domain, and overlays the
        annotation segments, for the range(start_sample, end_sample)

        Args:
            signal_sr_tuple (Tuple[np.ndarray, int]): A wave file loaded
                in as a tuple of the signal array and its sample rate.
                This is commonly the return of librosa.load(), so that
                can be used inline here.
            segments (List) : list of annotation segments in a tuple
                format of (start_time, end_time)
            start_sample (int): plot start range
            end_sample (int): plot end range
        """
        signal, sr = signal_sr_tuple

        if start_sample is None:
            start_sample = 0

        if end_sample is None:
            end_sample = len(signal)

        annotation_signal = self.convert_segments_to_signal(len(signal), sr, segments)

        # convert back to seconds for x-axis
        duration_secs = len(signal) / sr
        time_axis = np.linspace(0, duration_secs, len(signal))

        # trim signal and annotation_signal to
        # range(start_sample, end_sample)

        # scale annotation to max signal amplitude
        # max_amplitude = np.max([signal[start_sample:end_sample].max(), 0.3])
        # annotation_signal *= max_amplitude
        annotation_signal *= 0.5

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 1.5), constrained_layout=True)

        ax.plot(
            time_axis[start_sample:end_sample],
            signal[start_sample:end_sample],
            color="blue",
            alpha=0.5,
        )
        ax.plot(
            time_axis[start_sample:end_sample],
            annotation_signal[start_sample:end_sample],
            color="red",
            alpha=0.5,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Magnitude/Segments")
        # ax.set_ylim(-max_amplitude, max_amplitude)
        ax.set_ylim(-1, 1)
        ax.set_title(f"VAD Ground Truth {label}")
        plt.show()

        del signal
        del sr

        return None
