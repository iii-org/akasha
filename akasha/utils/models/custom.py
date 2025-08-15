from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field
from langchain.schema.embeddings import Embeddings
from langchain.llms.base import LLM


class custom_embed(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    client: Any = Field(default=None)  #: :meta private:
    model_name: str = "custom:custom-embedding-model"
    """Model name to use."""
    # model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = {}
    """Keyword arguments to pass when calling the `encode` method of the model."""

    def __init__(self, func: Any, encode_kwargs: Dict[str, Any] = {}, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        self.client = func
        self.encode_kwargs = encode_kwargs
        self.model_name = f"custom:{func.__name__}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        import numpy

        texts = list(map(lambda x: x.replace("\n", " "), texts))

        embeddings = self.client(texts, **self.encode_kwargs)

        if isinstance(embeddings, numpy.ndarray):
            embeddings = embeddings.tolist()

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]


class custom_model(LLM):
    max_token: int = 4096
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    tokenizer: Any = Field(default=None)
    model: Any = Field(default=None)
    func: Any = Field(default=None)

    def __init__(self, func: Callable, temperature: float = 0.001):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        super().__init__()
        self.func = func
        self.temperature = temperature
        if self.temperature == 0.0:
            self.temperature = 0.001

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return "custom: " + self.func.__name__

    def _call(
        self, prompt: str, stop: Optional[List[str]] = None, verbose: bool = True
    ) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """

        response = self.func(prompt)
        if verbose:
            print(response, end="", flush=True)
        return response

    def get_num_tokens(self, text: str) -> int:
        """get number of tokens in the text

        Args:
            **text (str)**: input text

        Returns:
            int: number of tokens
        """
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(text))
        return num_tokens
