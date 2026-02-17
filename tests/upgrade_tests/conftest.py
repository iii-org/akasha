import pytest
import os
from dotenv import load_dotenv
from pathlib import Path

def pytest_configure(config):
    """
    Load environment variables from .env file in the same directory as this file.
    """
    env_paths = [
        Path(__file__).parent / ".env",
        Path(__file__).parents[1] / "tests" / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        # print(f"Loaded environment variables from {env_path}")
    else:
        # print(f"No .env file found at {env_path}")
        pass
