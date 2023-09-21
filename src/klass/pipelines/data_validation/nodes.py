"""
This is a boilerplate pipeline 'data_validation'
generated using Kedro 0.18.12
"""
from typing import Dict, Callable
import json

import soundfile as sf 

def format_check(wav_files: Dict[str, Callable]) -> Dict:
    non_wav_files = []
    non_mono_files = []
    non_16khz_files = []

    # loop through wav files
    for wav_file_name, soundfile_info_callable in wav_files.items():
        # if not wave file
        soundfile_info = soundfile_info_callable()
        if soundfile_info.format != "WAV":
            non_wav_files.append(wav_file_name)
        # if wave file
        else:
            target_sample_rate = 16000

            # if not mono
            if soundfile_info.channels > 1:
                non_mono_files.append(wav_file_name)

            # if not 16khz
            if soundfile_info.samplerate != target_sample_rate:
                non_16khz_files.append(wav_file_name)

    non_wav_files = sorted(non_wav_files)
    non_mono_files = sorted(non_mono_files)
    non_16khz_files = sorted(non_16khz_files)

    format_check_report = {
            "non_wav": non_wav_files,
            "non_mono": non_mono_files,
            "non_16khz": non_16khz_files,
        }
    
    json_contents = json.dumps(format_check_report)

    return [json_contents]