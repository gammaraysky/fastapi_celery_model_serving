import concurrent.futures
import multiprocessing
import os
import shutil
import subprocess

import librosa
import numpy as np

from src.vad.data_prep.annotations import Annotations


class SoxWrapper:
    def __init__(
        self,
        annotations: dict,
        far: bool = False,
        num_train_file: int = 1,
        num_validation_file: int = 1,
        num_test_file: int = 1,
        output_dir: str = "",
        durations: float = 0.63,
        hard_limit: float = None,
    ):
        if far:
            self.lvl_1_keys = [_ for _ in list(annotations.keys()) if "far" in _]
        else:
            self.lvl_1_keys = [_ for _ in list(annotations.keys()) if "near" in _]
        self.annotations = annotations
        self.num_train_file = num_train_file
        self.num_validation_file = num_validation_file
        self.num_test_file = num_test_file
        self.train_val_test_meta_info = self._meta_data_of_files()
        self.output_dir = output_dir
        self.durations = durations
        self.hard_limit = hard_limit

    def segmentation_loader(self):
        file_names_in_list_form = self._create_copies_before_chop()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self._sox_chopping, file_names_in_list_form)

    def _meta_data_of_files(self):
        sox_digestable_format = {}
        for key in self.lvl_1_keys:
            if "train" in key:
                sox_digestable_format[key] = self._create_list_of_list_for_meta_data(
                    key, self.num_train_file
                )
            if "validation" in key:
                sox_digestable_format[key] = self._create_list_of_list_for_meta_data(
                    key, self.num_validation_file
                )
            if "test" in key:
                sox_digestable_format[key] = self._create_list_of_list_for_meta_data(
                    key, self.num_test_file
                )
        return sox_digestable_format

    def _create_list_of_list_for_meta_data(self, key, num_of_files_to_use):
        num_train_dataset = len(self.annotations[key])
        chose_file_by_index = np.random.choice(
            num_train_dataset, num_of_files_to_use, replace=False
        ).tolist()
        digestable_format = np.array(
            [self.annotations[key][_] for _ in self.annotations[key]]
        )
        digestable_format = digestable_format[chose_file_by_index]
        return digestable_format

    def _sox_chopping(self, audio_file):
        audio_duration = librosa.get_duration(path=audio_file)
        if not self.hard_limit:
            number_of_snippets = int(audio_duration / self.durations)
        else:
            number_of_snippets = self.hard_limit / self.durations
        for snippets in range(int(number_of_snippets)):
            start_time = round((0.63 * snippets), 2)
            end_time = round((0.63 * (snippets + 1)), 2)
            name_of_output_audio_file = (
                os.path.basename(audio_file).split(".wav")[0]
                + f"__{start_time}-{end_time}.wav"
            )
            root_path = os.path.dirname(audio_file) + "_trimmed"
            output_file = os.path.join(root_path, name_of_output_audio_file)
            with open("augmented_files.txt", "a") as w:
                w.write(f"{audio_file}    ->    {output_file}")
            command = [
                "sox",
                audio_file,
                output_file,
                "trim",
                str(start_time),
                str(end_time - start_time),
            ]
            subprocess.run(command)

    # def _create_copies_before_chop(self):
    #     meta_info_keys = self.train_val_test_meta_info.keys()
    #     new_audio_locatn = []
    #     for potential_folder_name in meta_info_keys:
    #         if not os.path.isdir(os.path.join(self.output_dir, potential_folder_name)):
    #             os.mkdir(os.path.join(self.output_dir, potential_folder_name))
    #         folder_name_for_trim_files = potential_folder_name + "_trimmed"
    #         if not os.path.isdir(os.path.join(self.output_dir, folder_name_for_trim_files)):
    #             os.mkdir(os.path.join(self.output_dir, folder_name_for_trim_files))
    #         output_towards = os.path.join(self.output_dir, potential_folder_name)
    #         meta_info_list = self.train_val_test_meta_info[potential_folder_name]
    #         [shutil.copy2(_["audio_path"], output_towards) for _ in meta_info_list]

    #     for _ in meta_info_keys:
    #         new_audio_locatn += [
    #             os.path.join(
    #                 self.output_dir, _, os.path.basename(meta_info["audio_path"])
    #             )
    #             for meta_info in self.train_val_test_meta_info[_]
    #         ]
    #     return new_audio_locatn

    def _copy_file(self, meta_info, potential_folder_name, output_dir):
        audio_path = meta_info["audio_path"]
        output_towards = os.path.join(output_dir, potential_folder_name)
        shutil.copy2(audio_path, output_towards)

    def _create_copies_before_chop(self):
        meta_info_keys = self.train_val_test_meta_info.keys()
        new_audio_locatn = []

        for potential_folder_name in meta_info_keys:
            if not os.path.isdir(os.path.join(self.output_dir, potential_folder_name)):
                os.mkdir(os.path.join(self.output_dir, potential_folder_name))
            folder_name_for_trim_files = potential_folder_name + "_trimmed"
            if not os.path.isdir(
                os.path.join(self.output_dir, folder_name_for_trim_files)
            ):
                os.mkdir(os.path.join(self.output_dir, folder_name_for_trim_files))
            # output_towards = os.path.join(self.output_dir, potential_folder_name)
            meta_info_list = self.train_val_test_meta_info[potential_folder_name]

            with multiprocessing.Pool() as pool:
                pool.starmap(
                    self._copy_file,
                    [
                        (info, potential_folder_name, self.output_dir)
                        for info in meta_info_list
                    ],
                )

            for meta_info in meta_info_list:
                new_audio_locatn.append(
                    os.path.join(
                        self.output_dir,
                        potential_folder_name,
                        os.path.basename(meta_info["audio_path"]),
                    )
                )

        return new_audio_locatn
