# import os
# import sys
# from pathlib import Path
# import shutil
# import hashlib
# from collections import OrderedDict

# import torch
# from torch_audiomentations import (
#     Compose,
#     Gain,
#     AddBackgroundNoise,
#     ApplyImpulseResponse,
#     HighPassFilter,
#     LowPassFilter,
# )
# import soundfile as sf
# import numpy as np


# def audio_file_name_hasher(filename):
#     """Convert a filename to a 256-bit hexadecimal string using SHA-256.

#     This function takes an audio file name as input and converts it into
#     a 256-bit hexadecimal string using the SHA-256 hashing algorithm.

#     Args:
#         filename (str): The audio file name in string format.

#     Returns:
#         str: A hexadecimal string representing the SHA-256 hash of the
#             filename.

#     Example:
#         To obtain a unique identifier for an audio file based on its
#         name, call this function with the filename as the argument. The
#         returned hexadecimal string can be used for various purposes,
#         such as data indexing or verification.
#     """
#     sha256_hash = hashlib.sha256()
#     sha256_hash.update(filename.encode("utf-8"))

#     return sha256_hash.hexdigest()


# def get_waveforms(audio_path: str):
#     """
#     Reads an audio file and returns the audio signal and its sample rate.

#     Args:
#         audio_path (str): The path to the audio file to be read.

#     Returns:
#         tuple: A tuple containing two elements:
#             - signal (numpy.ndarray): The audio signal as a NumPy array.
#             - samplerate (int): The sample rate of the audio.

#     Raises:
#         RuntimeError: If there is an error reading the audio file.

#     Note:
#         This function is a wrapper for the soundfile.read function.
#     """
#     signal, samplerate = sf.read(audio_path)
#     return signal, samplerate


# def generate_augmentations(signal, sample_rate):
#     '''
#     Generates augmented audio data from the input audio signal.

#     Args:
#         signal (numpy.ndarray): The input audio signal as a NumPy array.
#         sample_rate (int): The sample rate of the input audio signal.

#     Returns:
#         torch.Tensor: Augmented audio data as a 1D PyTorch tensor.

#     Note:
#         This function applies augmentations to the input audio signal using the
#         `apply_augmentation` function and reshapes the result into a 1D tensor.
#     '''

#     return torch.reshape(
#         apply_augmentation(torch.Tensor(signal), sample_rate=sample_rate), (-1,)
#     )


# def main():
#     """
#     Script augments audio wav files when given an input and output
#     directory and a probability of augmentation as command line arguments.
#     Args:
#          root (str/path): audio wav files directory that requires augmentation
#          root_rttm (str/path): rttm files directory corresponding to the audio wav files found in root directory
#          output_path (str/path): output directory to write new augmented wav and rttm files, ensure that directory currently exists with /audio and /rttm sub paths
#          prob_of_augmentation (float): Probability of augmentation (eg. 0.3 will augment 30% of files in root directory)
#     Returns:
#          NA
#     """
#     root = sys.argv[1]
#     root_rttm = sys.argv[2]
#     output_path = sys.argv[3]
#     prob_of_augmentation = float(sys.argv[4])
#     assert (
#         prob_of_augmentation >= 0 and prob_of_augmentation <= 1
#     ), "prob_of_augmentation variable should be between 0 and 1 inclusive"

#     count = 0
#     list_of_path = []

#     torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for path in Path(root).rglob("*.wav"):
#         if path not in list_of_path:
#             count += 1
#             list_of_path.append(path)

#     print("Number of wav files: {}".format(count))

#     total_no_of_audio_files = len(list_of_path)
#     no_of_audio_files_to_augment = int(
#         prob_of_augmentation * total_no_of_audio_files
#     )  # rounded to nearest integer

#     if float(prob_of_augmentation) != 0.0 and no_of_audio_files_to_augment != 0:
#         dict_of_hashed_name = {}

#         for path in list_of_path:
#             filename = os.path.basename(path)
#             hashed_name = audio_file_name_hasher(filename)
#             if hashed_name not in dict_of_hashed_name:
#                 dict_of_hashed_name[hashed_name] = path

#         sorted_dict_of_hashes = OrderedDict(sorted(dict_of_hashed_name.items()))

#         no_of_audio_augmented = 0
#         list_of_dict_items = list(sorted_dict_of_hashes.items())
#         for item in list_of_dict_items:
#             key = item[0]
#             value = item[1]
#             signal, samplerate = get_waveforms(value)
#             signal = np.reshape(signal, (1, 1, -1))

#             augmented_signal_array = generate_augmentations(signal, samplerate)

#             sf.write(
#                 f"{output_path}/audio/{os.path.basename(value)}",
#                 augmented_signal_array,
#                 samplerate,
#             )
#             rttm_file_name = os.path.basename(value).split(".wav")[0] + ".rttm"

#             try:
#                 shutil.copy2(
#                     f"{root_rttm}/{rttm_file_name}",
#                     f"{output_path}/rttm/{rttm_file_name}",
#                 )
#             except FileNotFoundError:
#                 print(f"{os.path.basename(value)}'s rttm file not found")

#             no_of_audio_augmented += 1
#             del sorted_dict_of_hashes[key]

#             if no_of_audio_augmented == no_of_audio_files_to_augment:
#                 break

#         # copying the remaining unaugmented audio
#         if len(list(sorted_dict_of_hashes.keys())) != 0:
#             for item in sorted_dict_of_hashes.keys():
#                 shutil.copy2(f"{item}", f"{output_path}/rttm/{os.path.basename(item)}")
#                 rttm_file_name = os.path.basename(item).split(".wav")[0] + ".rttm"
#                 shutil.copy2(
#                     f"{root_rttm}/{rttm_file_name}",
#                     f"{output_path}/rttm/{rttm_file_name}",
#                 )

#     else:
#         print("prob_of_augmentation set to 0.0, no augmentation required")


# if __name__ == "__main__":
#     main()
