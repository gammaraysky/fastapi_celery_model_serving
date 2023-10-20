"""Utility to generate synthetic wave file and rttm file. Useful for
testing.
"""
import soundfile as sf
import numpy as np
import argparse
from pathlib import Path


def save_wavfile(filename, duration, sample_rate):
    # Create a NumPy array of zeros for the audio signal
    num_samples = int(duration * sample_rate)
    audio_signal = np.zeros(num_samples, dtype=np.float32)

    # Save the audio signal to the specified WAV file
    sf.write(filename, audio_signal, sample_rate)

    print(f"Saved {duration}-second silence to {filename}")


def save_rttmfile(filename):
    with open(filename, "w", encoding="utf8") as rttmfile:
        rttmfile.write("""SPEAKER file1 1 1.0 5.0 <NA> <NA> SPEECH <NA> <NA>""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a silent WAV file.")
    parser.add_argument("filename", type=str, help="Output WAV file name")
    parser.add_argument(
        "--duration", type=float, default=10, help="Duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)",
    )

    args = parser.parse_args()

    save_wavfile(args.filename, args.duration, args.sample_rate)
    save_rttmfile(str(Path(args.filename).with_suffix(".rttm")))
