from setuptools import setup

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

install_requires = [
    "pypdf",
    "langchain-core>=0.3,<0.4",
    "langchain>=0.3,<0.4",
    "langchain-community>=0.3,<0.4",
    "langchain_openai>=0.1.0",
    "langchain-huggingface>=0.1.2",
    "langchain-chroma==0.2.2",
    "chromadb==0.4.14",
    "onnxruntime <= 1.20.0",
    "openai>=1.78.1",
    "tiktoken",
    "scikit-learn>=1.3.0",
    "jieba>=0.42.1",
    "opencc-python-reimplemented==0.1.7",
    "sentence-transformers>=3.1.1",
    "transformers>=4.50.0",  # ==4.41.1
    "accelerate>=1.7.0",
    "tqdm>=4.65.0",
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
    "wikipedia",
    "numpy<2",
    "sentencepiece",
    "google-genai>=1.15.0",
    "anthropic",
    "tavily-python",
    "faiss-cpu",
    "langchain-mcp-adapters>=0.1.1",
    "mlflow",
    "setuptools",
]

### install different torch version###
install_requires.append("torch==2.2.0; platform_system=='Windows'")
install_requires.append("torch==2.0.1; platform_system=='Darwin'")
install_requires.append("torch==2.2.0; platform_system=='Linux'")

install_requires.append("torchvision==0.17.0; platform_system=='Windows'")
install_requires.append("torchvision==0.15.2; platform_system=='Darwin'")
install_requires.append("torchvision==0.17.0; platform_system=='Linux'")
setup(
    name="akasha_terminal",
    version="0.9.09",
    description="document QA(RAG) package using langchain and chromadb",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="chih chuan chang",
    url="https://github.com/iii-org/akasha",
    author_email="ccchang@iii.org.tw",
    install_requires=install_requires,
    extras_require={
        "llama-cpp": [
            "llama-cpp-python>=0.3.8",
        ],
        "peft": ["auto-gptq==0.3.1"],
    },
    packages=[
        "akasha",
        "akasha.agent",
        "akasha.interface",
        "cli",
        "akasha.eval",
        "akasha.helper",
        "akasha.RAG",
        "akasha.tools",
        "akasha.utils",
        "akasha.utils.models",
        "akasha.utils.db",
        "akasha.utils.prompts",
        "akasha.utils.search",
        "akasha.utils.search.retrievers",
    ],
    entry_points={"console_scripts": ["akasha = cli.glue:akasha"]},
    python_requires=">=3.10,<3.13",
)
