from charset_normalizer import detect
from typing import Union
from pathlib import Path
import uuid
import hashlib


def detect_encoding(file_path: Union[str, Path]) -> str:
    with open(file_path, "rb") as file:
        raw_data = file.read(1000)  # Read a portion of the file
        result = detect(raw_data)
        return result["encoding"]


def get_text_md5(text):
    md5_hash = hashlib.md5(text.encode()).hexdigest()

    return md5_hash


def get_mac_address() -> str:
    # Get the MAC address
    mac = uuid.getnode()
    # Convert the MAC address to a readable format without colons
    mac_address = "".join(
        ["{:02x}".format((mac >> elements) & 0xFF) for elements in range(0, 2 * 6, 2)][
            ::-1
        ]
    )
    return get_text_md5(mac_address)
