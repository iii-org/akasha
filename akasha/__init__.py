import pathlib
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
load_dotenv(pathlib.Path().cwd() / ".env")

from .RAG import RAG

__all__ = ['RAG']
