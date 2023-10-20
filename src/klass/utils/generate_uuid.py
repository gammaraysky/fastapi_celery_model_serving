"""
Add 'uuid' and 'timestamp' keys to a model checkpoint file
"""
import time
import uuid

import torch


def add_uuid_and_timestamp(ckpt_path: str):
    """Add 'uuid' and 'timestamp' keys to a model checkpoint file

    Generates a UUID string and current timestamp and saves them as
    key-values to a PyanNet model checkpoint file.

    Args:
        ckpt_path (str): Path to model checkpoint file

    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    checkpoint["uuid"] = str(uuid.uuid4())
    checkpoint["timestamp"] = time.time()
    # print(checkpoint.keys())
    torch.save(checkpoint, ckpt_path)


# add_uuid_and_timestamp(
#     "C:/Users/Han/AIAP/KLASS/dev/190/vad/vol_mount/model_checkpoints/epoch=20.ckpt"
# )
