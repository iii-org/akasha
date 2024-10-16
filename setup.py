from setuptools import setup
import platform, os, subprocess
from setuptools.command.install import install
from setuptools.command.develop import develop
# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = "n"  #(this_directory / "read.md").read_text(encoding="utf-8")

install_requires = [
    "pypdf",
    "langchain>=0.1.0,<=0.1.16",
    "langchain_openai>=0.1.0",
    "chromadb==0.4.14",
    "openai>=0.27",
    "tiktoken",
    "scikit-learn<1.3.0",
    "jieba>=0.42.1",
    "sentence-transformers>=3.1.1",
    "torch==2.2.0",
    "transformers>=4.45.0",  #==4.41.1
    "auto-gptq==0.3.1",
    "tqdm==4.65.0",
    "docx2txt==0.8",
    "rouge==1.0.1",
    "rouge-chinese==1.0.3",
    "bert-score==0.3.13",
    "click",
    "tokenizers>=0.19.1",
    "streamlit>=1.33.0",
    "streamlit_option_menu>=0.3.6",
    "rank_bm25",
    "unstructured",
    "python-pptx",
    "wikipedia"
]
# if platform.system() == "Windows":
#     install_requires.append("opencc==1.1.1")
# elif platform.system() == "Darwin":
#     install_requires.append("opencc==0.2")
# else:
#     install_requires.append("opencc==1.1.6")

install_requires.append("opencc==1.1.1; platform_system=='Windows'")
install_requires.append("opencc==0.2; platform_system=='Darwin'")
install_requires.append("opencc==1.1.6; platform_system=='Linux'")


class CustomDevelopCommand(develop):
    """Custom handler for the 'develop' command."""

    def run(self):
        if 'llama-cpp' in self.distribution.extras_require:
            self._install_llama_cpp()
        develop.run(self)

    def _install_llama_cpp(self):
        os.environ['CMAKE_ARGS'] = "-DGGML_CUDA=on"
        os.environ['FORCE_CMAKE'] = "1"
        subprocess.check_call([
            "python", "-m", "pip", "install", "--upgrade", "--force-reinstall",
            "llama-cpp-python>=0.2.6", "--no-cache-dir"
        ])


class CustomInstallCommand(install):
    """Custom handler for the 'install' command."""

    def run(self):
        if 'llama-cpp' in self.distribution.extras_require:
            self._install_llama_cpp()
        install.run(self)

    def _install_llama_cpp(self):
        os.environ['CMAKE_ARGS'] = "-DGGML_CUDA=on"
        os.environ['FORCE_CMAKE'] = "1"
        subprocess.check_call([
            "python", "-m", "pip", "install", "--upgrade", "--force-reinstall",
            "llama-cpp-python>=0.2.6", "--no-cache-dir"
        ])


setup(
    name="akasha-terminal",
    version="0.8.63",
    description="document QA(RAG) package using langchain and chromadb",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="chih chuan chang",
    url="https://github.com/iii-org/akasha",
    author_email="ccchang@iii.org.tw",
    install_requires=install_requires,
    extras_require={
        'llama-cpp-gpu': [
            "llama-cpp-python>=0.2.6",
        ],
        'llama-cpp': [
            "llama-cpp-python>=0.2.6",
        ]
    },
    packages=[
        "akasha",
        "akasha.models",
        "cli",
        "akasha.eval",
        #    "akasha.interface",
    ],
    entry_points={"console_scripts": ["akasha = cli.glue:akasha"]},
    python_requires=">=3.8",
    cmdclass={
        'install': CustomInstallCommand,  # Override install command
        'develop':
        CustomDevelopCommand,  # Override develop command for development installs
    })
