from pathlib import Path

from dotenv import load_dotenv


def pytest_configure():
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
