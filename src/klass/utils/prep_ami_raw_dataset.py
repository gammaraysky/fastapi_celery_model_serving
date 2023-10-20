"""
AMI Raw Dataset Preparation
---------------------------
This is pre-configured to accept original AMI folder structure and XML
files.

Set the paths to the original folders as needed in
conf/base/prep_raw_data.yaml.

It will make copies and reorganize the files into the folder format
required for our pipeline.

which will be as follows:
data/interim/<dataset>/<split>/audio
                              /xml
                              /textgrid
                              /rttm

Original XML files (which are only for near field) will be:
1) converted to RTTM files and renamed to match the near field wave
   files.
2) combined and merged to create annotations for the far field, and also
   saved as RTTM with filenames matching the far field audio.
"""

import logging
import os
import re
import sys
from pathlib import Path

import hydra
from tqdm import tqdm

sys.path.append(os.getcwd())

import src.klass.utils.general_utils as genutils
from src.klass.utils.data_prep import speech_segments as ss
from src.klass.utils.data_prep.prepdatafolders import AmiPrepDataFolders


@hydra.main(config_path="../../../conf/base", config_name="prep_raw_data.yaml")
def main(args):
    """AMI Raw Dataset Preparation
    ---------------------------
    This is pre-configured to accept original AMI folder structure and XML
    files.

    Set the paths to the original folders as needed in
    conf/base/prep_raw_data.yaml.

    It will make copies and reorganize the files into the folder format
    required for our pipeline.

    which will be as follows:
    data/interim/<dataset>/<split>/audio
                                /xml
                                /textgrid
                                /rttm

    Original XML files (which are only for near field) will be:
    1) converted to RTTM files and renamed to match the near field wave
    files.
    2) combined and merged to create annotations for the far field, and also
    saved as RTTM with filenames matching the far field audio.
    """

    # SETUP LOGGER
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/base/logging.yml"
    )
    genutils.setup_logging(logger_config_path)

    # SETUP PATHS
    data_basepath = Path(args["data_paths"]["basepath"])
    ami_raw_near_audio_subpath = Path(
        args["data_paths"]["prepdatafolders"]["ami_raw"]["near_audio_subpath"]
    )
    ami_raw_far_audio_subpath = Path(
        args["data_paths"]["prepdatafolders"]["ami_raw"]["far_audio_subpath"]
    )
    ami_raw_near_xml_subpath = Path(
        args["data_paths"]["prepdatafolders"]["ami_raw"]["near_xml_subpath"]
    )

    ami_raw_near_audio_path = data_basepath.joinpath(ami_raw_near_audio_subpath)
    ami_raw_far_audio_path = data_basepath.joinpath(ami_raw_far_audio_subpath)
    ami_raw_near_xmls_path = data_basepath.joinpath(ami_raw_near_xml_subpath)

    ami_far_path = Path(
        args["data_paths"]["prepdatafolders"]["ami_prepped"]["far_path"]
    )
    ami_far_path = data_basepath.joinpath(ami_far_path)

    ami_near_path = Path(
        args["data_paths"]["prepdatafolders"]["ami_prepped"]["near_path"]
    )
    ami_near_path = data_basepath.joinpath(ami_near_path)

    data_splits = args["prepdatafolders"]["ami_raw"]["data_splits"]

    ami = AmiPrepDataFolders(data_splits)

    ############################################
    ### * FIND AND COPY AMI NEAR AUDIO FILES ###
    logger.info("Collating files from original dataset...")
    ami_near_files = ami.find_files_in_folder(".wav", ami_raw_near_audio_path)
    (
        ami_near_train_files,
        ami_near_val_files,
        ami_near_test_files,
    ) = ami.train_val_test_split(ami_near_files)

    logger.info("Copying to destination path/folder structure...")
    ami_near_train_files = ami.copyfiles(
        ami_near_train_files, ami_near_path.joinpath("train/audio")
    )
    ami_near_val_files = ami.copyfiles(
        ami_near_val_files, ami_near_path.joinpath("val/audio")
    )
    ami_near_test_files = ami.copyfiles(
        ami_near_test_files, ami_near_path.joinpath("test/audio")
    )

    logger.info("Copied AMI near audio files into %s", str(ami_near_path))

    ###########################################
    ### * FIND AND COPY AMI FAR AUDIO FILES ###
    ami_far_files = ami.find_files_in_folder(".wav", ami_raw_far_audio_path)
    # train test split the original file list according to pre-defined splits in dataset
    # dictionary
    (
        ami_far_train_files,
        ami_far_val_files,
        ami_far_test_files,
    ) = ami.train_val_test_split(ami_far_files)

    # copy ami far files into correct folder structure
    ami_far_train_files = ami.copyfiles(
        ami_far_train_files, ami_far_path.joinpath("train/audio")
    )
    ami_far_val_files = ami.copyfiles(
        ami_far_val_files, ami_far_path.joinpath("val/audio")
    )
    ami_far_test_files = ami.copyfiles(
        ami_far_test_files, ami_far_path.joinpath("test/audio")
    )
    logger.info("Copied AMI far audio files into %s", str(ami_far_path))

    ##########################################
    ### * FIND AND COPY AMI NEAR XML FILES ###
    # grab all ami nearfield xmls and train/test/split by dataset definitions
    ami_near_xmlpaths = ami.find_files_in_folder(".xml", ami_raw_near_xmls_path)
    # train test split the original file list according to pre-defined splits in dataset
    # dictionary
    (
        ami_near_train_xmlpaths,
        ami_near_val_xmlpaths,
        ami_near_test_xmlpaths,
    ) = ami.train_val_test_split(ami_near_xmlpaths)

    # copy ami near xmls files into correct folder structure
    ami_near_train_xmlpaths = ami.copyfiles(
        ami_near_train_xmlpaths, ami_near_path.joinpath("train/xml")
    )
    ami_near_val_xmlpaths = ami.copyfiles(
        ami_near_val_xmlpaths, ami_near_path.joinpath("val/xml")
    )
    ami_near_test_xmlpaths = ami.copyfiles(
        ami_near_test_xmlpaths, ami_near_path.joinpath("test/xml")
    )
    logger.info("Copied AMI near XML files into %s", str(ami_near_path))

    ########################################################
    ### * RENAME AMI NEAR XML FILES TO MATCH AUDIO FILES ###
    # Currently: EN2001a.A.segments.xml   EN2001a.Headset-0.wav
    # Rename to: EN2001a.Headset-0.xml    EN2001a.Headset-0.wav
    logger.info("Renaming AMI near XML files to match AMI audio files...")
    ami_near_train_xmlpaths = ami.rename_xmlfiles(ami_near_train_xmlpaths)
    ami_near_val_xmlpaths = ami.rename_xmlfiles(ami_near_val_xmlpaths)
    ami_near_test_xmlpaths = ami.rename_xmlfiles(ami_near_test_xmlpaths)
    logger.info("Done")

    ########################################
    ### * CONVERT AMI NEAR XMLS TO RTTMS ###
    # read xml as segments
    logger.info("Converting AMI near XML files to RTTMs...")

    ami_near_train_segments = [
        ss.read_xml(xml_filepath) for xml_filepath in ami_near_train_xmlpaths
    ]
    ami_near_val_segments = [
        ss.read_xml(xml_filepath) for xml_filepath in ami_near_val_xmlpaths
    ]
    ami_near_test_segments = [
        ss.read_xml(xml_filepath) for xml_filepath in ami_near_test_xmlpaths
    ]

    # save as rttm
    for ami_segment, ami_xmlpath in zip(
        ami_near_train_segments + ami_near_val_segments + ami_near_test_segments,
        ami_near_train_xmlpaths + ami_near_val_xmlpaths + ami_near_test_xmlpaths,
    ):
        ss.write_rttm(
            ami_segment,
            ami.replace_path(
                old_path=ami_xmlpath, new_subfolder="rttm", new_ext=".rttm"
            ),
            file_id=Path(ami_xmlpath).stem,
        )
    logger.info("Done")

    ####################################################################################
    ### * FIND FILENAME MATCHES AMI FAR FIELD WAVE FILES TO AMI NEAR FIELD SEGMENTS. ###
    # there will be one to many matching. 1 far field audio comprises several speaker
    # segments of the near field audio. for each set of matches, combine the XMLs.

    # get front part of filename i.e 'ES2005b.Array1-01.wav' -> only grab 'ES2005b'
    logger.info("Generate AMI Far Field RTTMs by merging Near Field RTTMs...")

    ami_far_train_stems = [
        str(fpath.stem).split(".")[0] for fpath in ami_far_train_files
    ]
    ami_far_val_stems = [str(fpath.stem).split(".")[0] for fpath in ami_far_val_files]
    ami_far_test_stems = [str(fpath.stem).split(".")[0] for fpath in ami_far_test_files]

    ami_far_train_stems = set(ami_far_train_stems)
    ami_far_val_stems = set(ami_far_val_stems)
    ami_far_test_stems = set(ami_far_test_stems)

    for file_id in tqdm(ami_far_train_stems):
        # find several matches of file_id in XMLs:
        # e.g. EN2001a.Headset-0.xml, EN2001a.Headset-1.xml, EN2001a.Headset-2.xml
        xml_matches = [
            item for item in ami_near_train_xmlpaths if re.search(file_id, item)
        ]

        # find several matches of file_id in Far Field audio files:
        # e.g. EN2001a.Array1-01.wav, EN2001a.Array1-02.wav, EN2001a.Array1-03.wav...
        wav_matches = [
            item for item in ami_far_train_files if re.search(file_id, str(item))
        ]

        # combined segments across several speakers
        segments = ss.merge_xmls_to_one_segment(xml_matches)

        # for each Array1-01 to Array1-08, save the same RTTM segments
        for wav_filename in wav_matches:
            # replace name and save rttm
            output_rttm_path = ami.replace_path(
                old_path=wav_filename, new_subfolder="rttm", new_ext=".rttm"
            )

            # save rttm
            ss.write_rttm(segments, output_rttm_path, file_id=Path(wav_filename).stem)

    for file_id in tqdm(ami_far_val_stems):
        # find several matches of file_id in XMLs:
        # e.g. EN2001a.Headset-0.xml, EN2001a.Headset-1.xml, EN2001a.Headset-2.xml
        xml_matches = [
            item for item in ami_near_val_xmlpaths if re.search(file_id, item)
        ]

        # find several matches of file_id in Far Field audio files:
        # e.g. EN2001a.Array1-01.wav, EN2001a.Array1-02.wav, EN2001a.Array1-03.wav...
        wav_matches = [
            item for item in ami_far_val_files if re.search(file_id, str(item))
        ]

        # combined segments across several speakers
        segments = ss.merge_xmls_to_one_segment(xml_matches)

        # for each Array1-01 to Array1-08, save the same RTTM segments
        for wav_filename in wav_matches:
            # replace name and save rttm
            output_rttm_path = ami.replace_path(
                old_path=wav_filename, new_subfolder="rttm", new_ext=".rttm"
            )

            # save rttm
            ss.write_rttm(segments, output_rttm_path, file_id=Path(wav_filename).stem)

    for file_id in tqdm(ami_far_test_stems):
        # find several matches of file_id in XMLs:
        # e.g. EN2001a.Headset-0.xml, EN2001a.Headset-1.xml, EN2001a.Headset-2.xml
        xml_matches = [
            item for item in ami_near_test_xmlpaths if re.search(file_id, item)
        ]

        # find several matches of file_id in Far Field audio files:
        # e.g. EN2001a.Array1-01.wav, EN2001a.Array1-02.wav, EN2001a.Array1-03.wav...
        wav_matches = [
            item for item in ami_far_test_files if re.search(file_id, str(item))
        ]

        # combined segments across several speakers
        segments = ss.merge_xmls_to_one_segment(xml_matches)

        # for each Array1-01 to Array1-08, save the same RTTM segments
        for wav_filename in wav_matches:
            # replace name and save rttm
            output_rttm_path = ami.replace_path(
                old_path=wav_filename, new_subfolder="rttm", new_ext=".rttm"
            )

            # save rttm
            ss.write_rttm(segments, output_rttm_path, file_id=Path(wav_filename).stem)

    logger.info("Done")


if __name__ == "__main__":
    main()
