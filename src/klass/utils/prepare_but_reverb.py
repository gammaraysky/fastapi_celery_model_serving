""" For collecting all BUT Reverb impulse response wave files into a
single folder, and renaming each file to contain the metadata details
(originally each file is only named
IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav in their respective sub paths
and the sub paths contain the metadata e.g. which location, which mic..)

These room impulse responses are used in the data augmentation pipeline,
to augment train samples during model training.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.append(Path(os.getcwd()).parent)
from src.klass.utils.data_prep.prepdatafolders import PrepDataFolders

src_folder = "../../data/but_reverb"
dest_folder = "../../data/BUTreverb_rirs/"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


prep = PrepDataFolders()

# find all .wav files from BUT Reverb dataset
rirs = prep.find_files_in_folder(ext=".wav", folderpath=src_folder, recursive=True)

# drop files containing 'silence'
rirs = [rir for rir in rirs if "silence" not in str(rir)]

Path(dest_folder).mkdir(parents=True, exist_ok=True)

# copy into our folder, renaming each file to have unique filename based
# on the dataset's file subpaths
for rir in rirs:
    new_fname = "-".join(str(rir).split("/")[-6:-2] + str(rir).split("/")[-1:])
    shutil.copy2(rir, Path(dest_folder).joinpath(new_fname))
    logger.info(new_fname)
