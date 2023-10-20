"""
Alimeeting Raw Dataset Preparation
----------------------------------
This is pre-configured to accept original Alimeeting folder structure
and TextGrid files.

Set the paths to the original folders as needed in
conf/base/prep_raw_data.yaml.

It will make copies and reorganize the files into the folder format
required for our pipeline.

which will be as follows:
data/interim/<dataset>/<split>/audio
                              /xml
                              /textgrid
                              /rttm

It will also split Ali Far 8ch wave files into separate mono files.

Original TextGrid files will be converted to RTTM files.
"""
import logging
import os
import sys
from pathlib import Path

import hydra
from tqdm import tqdm

sys.path.append(os.getcwd())

import src.klass.utils.general_utils as genutils
from src.klass.utils.data_prep import speech_segments as ss
from src.klass.utils.data_prep.prepdatafolders import AliPrepDataFolders


@hydra.main(config_path="../../../conf/base", config_name="prep_raw_data.yaml")
def main(args):
    """Alimeeting Raw Dataset Preparation
    ----------------------------------
    This is pre-configured to accept original Alimeeting folder structure
    and TextGrid files.

    Set the paths to the original folders as needed in
    conf/base/prep_raw_data.yaml.

    It will make copies and reorganize the files into the folder format
    required for our pipeline.

    which will be as follows:
    data/interim/<dataset>/<split>/audio
                                /xml
                                /textgrid
                                /rttm

    It will also split Ali Far 8ch wave files into separate mono files.

    Original TextGrid files will be converted to RTTM files.
    """

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/base/logging.yml"
    )
    genutils.setup_logging(logger_config_path)

    ali = AliPrepDataFolders()

    #####################
    ### * SETUP PATHS ###
    data_basepath = Path(args["data_paths"]["basepath"])

    ali_raw_far_train_audio = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_raw"]["far_train_audio"])
    )
    ali_raw_far_train_tgrid = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_raw"]["far_train_tgrid"])
    )
    ali_raw_near_train_audio = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_raw"]["near_train_audio"])
    )
    ali_raw_near_train_tgrid = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_raw"]["near_train_tgrid"])
    )
    ali_raw_far_val_audio = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_raw"]["far_val_audio"])
    )
    ali_raw_far_val_tgrid = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_raw"]["far_val_tgrid"])
    )
    ali_raw_near_val_audio = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_raw"]["near_val_audio"])
    )
    ali_raw_near_val_tgrid = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_raw"]["near_val_tgrid"])
    )
    ali_raw_far_test_audio = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_raw"]["far_test_audio"])
    )
    ali_raw_far_test_tgrid = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_raw"]["far_test_tgrid"])
    )
    ali_raw_near_test_audio = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_raw"]["near_test_audio"])
    )
    ali_raw_near_test_tgrid = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_raw"]["near_test_tgrid"])
    )

    ali_near_path = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_prepped"]["near_path"])
    )
    ali_far_path = data_basepath.joinpath(
        Path(args["data_paths"]["prepdatafolders"]["ali_prepped"]["far_path"])
    )

    ###############################################################################
    ### * GRAB AUDIO AND TEXTGRID FILES FOR EACH SUBSET (NEAR VS FAR) AND SPLIT ###
    logger.info("Collating files from original dataset...")
    ali_raw_far_train_audiopaths = ali.find_files_in_folder(
        ".wav", ali_raw_far_train_audio, recursive=False
    )
    ali_raw_far_train_tgridpaths = ali.find_files_in_folder(
        ".TextGrid", ali_raw_far_train_tgrid, recursive=False
    )
    ali_raw_near_train_audiopaths = ali.find_files_in_folder(
        ".wav", ali_raw_near_train_audio, recursive=False
    )
    ali_raw_near_train_tgridpaths = ali.find_files_in_folder(
        ".TextGrid", ali_raw_near_train_tgrid, recursive=False
    )
    ali_raw_far_val_audiopaths = ali.find_files_in_folder(
        ".wav", ali_raw_far_val_audio, recursive=False
    )
    ali_raw_far_val_tgridpaths = ali.find_files_in_folder(
        ".TextGrid", ali_raw_far_val_tgrid, recursive=False
    )
    ali_raw_near_val_audiopaths = ali.find_files_in_folder(
        ".wav", ali_raw_near_val_audio, recursive=False
    )
    ali_raw_near_val_tgridpaths = ali.find_files_in_folder(
        ".TextGrid", ali_raw_near_val_tgrid, recursive=False
    )
    ali_raw_far_test_audiopaths = ali.find_files_in_folder(
        ".wav", ali_raw_far_test_audio, recursive=False
    )
    ali_raw_far_test_tgridpaths = ali.find_files_in_folder(
        ".TextGrid", ali_raw_far_test_tgrid, recursive=False
    )
    ali_raw_near_test_audiopaths = ali.find_files_in_folder(
        ".wav", ali_raw_near_test_audio, recursive=False
    )
    ali_raw_near_test_tgridpaths = ali.find_files_in_folder(
        ".TextGrid", ali_raw_near_test_tgrid, recursive=False
    )

    #######################################################################
    ### * COPY AUDIO AND TEXTGRID FILES TO DESTINATION FOLDER STRUCTURE ###

    logger.info("Copying files to destination path/folder structure...")
    ali_far_train_audiopaths = ali.copyfiles(
        ali_raw_far_train_audiopaths,
        ali_far_path.joinpath("train/audio"),
    )
    ali_far_train_tgridpaths = ali.copyfiles(
        ali_raw_far_train_tgridpaths,
        ali_far_path.joinpath("train/textgrid"),
    )
    ali_far_val_audiopaths = ali.copyfiles(
        ali_raw_far_val_audiopaths, ali_far_path.joinpath("val/audio")
    )
    ali_far_val_tgridpaths = ali.copyfiles(
        ali_raw_far_val_tgridpaths,
        ali_far_path.joinpath("val/textgrid"),
    )
    ali_far_test_audiopaths = ali.copyfiles(
        ali_raw_far_test_audiopaths,
        ali_far_path.joinpath("test/audio"),
    )
    ali_far_test_tgridpaths = ali.copyfiles(
        ali_raw_far_test_tgridpaths,
        ali_far_path.joinpath("test/textgrid"),
    )

    logger.info("Copied Ali Far audio & textgrid files to %s", str(ali_far_path))

    ali_near_train_audiopaths = ali.copyfiles(
        ali_raw_near_train_audiopaths,
        ali_near_path.joinpath("train/audio"),
    )
    ali_near_train_tgridpaths = ali.copyfiles(
        ali_raw_near_train_tgridpaths,
        ali_near_path.joinpath("train/textgrid"),
    )
    ali_near_val_audiopaths = ali.copyfiles(
        ali_raw_near_val_audiopaths,
        ali_near_path.joinpath("val/audio"),
    )
    ali_near_val_tgridpaths = ali.copyfiles(
        ali_raw_near_val_tgridpaths,
        ali_near_path.joinpath("val/textgrid"),
    )
    ali_near_test_audiopaths = ali.copyfiles(
        ali_raw_near_test_audiopaths,
        ali_near_path.joinpath("test/audio"),
    )
    ali_near_test_tgridpaths = ali.copyfiles(
        ali_raw_near_test_tgridpaths,
        ali_near_path.joinpath("test/textgrid"),
    )

    logger.info("Copied Ali Near audio & textgrid files to %s", str(ali_near_path))

    logger.info("Converting original TextGrid annotations to RTTMs...")
    ####################################
    ### * CONVERT TEXTGRIDS TO RTTMS ###
    for textgridpath in tqdm(
        ali_far_train_tgridpaths
        + ali_near_train_tgridpaths
        + ali_far_val_tgridpaths
        + ali_near_val_tgridpaths
        + ali_far_test_tgridpaths
        + ali_near_test_tgridpaths
    ):
        speech_segments = ss.read_textgrid(textgridpath)

        speech_segments = ss.merge_overlap_segments([speech_segments])

        ss.write_rttm(
            speech_segments,
            ali.replace_path(
                old_path=textgridpath, new_subfolder="rttm", new_ext=".rttm"
            ),
            file_id=Path(textgridpath).stem,
        )

    logger.info("Done")

    ###################################################
    ### * RENAME ALI FAR RTTMS TO MATCH AUDIO FILES ###
    logger.info("Renaming Ali Far RTTM to match audio filenames...")
    ali_far_rttms_paths = ali.match_rename_far_rttms_to_audio(ali_far_path)
    logger.info("Done")

    #############################################################
    ### * UPDATE RENAMED ALI RTTM FILES TO HAVE FILE_ID MATCH ###
    logger.info("Updating Ali Far RTTM file_id labels...")
    ali.update_rttm_fileids(ali_far_path)
    logger.info("Done")

    ##################################################################
    ### * SPLIT ALI FAR 8CH WAV INTO 8 SEPARATE .WAVS (AND .RTTMS) ###
    logger.info("Split Ali Far 8ch into separate files...")
    ali.split_far_multichannel(ali_far_path)
    logger.info("Done")


if __name__ == "__main__":
    main()
