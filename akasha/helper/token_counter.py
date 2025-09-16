from pathlib import Path
import os

from akasha.helper.base import separate_name


class myTokenizer(object):
    """this class is for computing the number of tokens in a given text using different tokenizers.

    Args:
        object (_type_): _description_
    """

    def __init__(self, model_id: str, tokenizer: object, path: str = "./tokenizers"):
        """
        Initialize a Tokenizer object.

        Args:
            model_id (str): The name of the model for the tokenizer.
            tokenizer (object): The tokenizer object from HuggingFace.
            path (str, optional): The path to save the tokenizer. Defaults to './tokenizers'.
        """
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.path = path

    def compute_tokens_huggingface(self, text: str) -> int:
        """
        Compute the number of tokens in a given text using huggingface tokenizer.

        Args:
            text (str): The text to be tokenized.

        Returns:
            int: The number of tokens in the text.
        """
        tokens = self.tokenizer(text)
        num_tokens = len(tokens["input_ids"])
        return num_tokens

    @staticmethod
    def compute_tokens_anthropic(text: str, model_name: str) -> int:
        ### take too long time to load the model, skip for now ###
        import anthropic

        token_count = anthropic.Anthropic().beta.messages.count_tokens(
            model=model_name, messages=[{"role": "user", "content": text}]
        )

        return token_count.input_tokens

    @staticmethod
    def compute_tokens_openai(text: str, model_name: str) -> int:
        """
        Compute the number of tokens in a given text using OpenAI tiktoken.

        Args:
            text (str): The text to be tokenized.
            model_name (str): The name of the OpenAI model. ex. 'openai:gpt-3.5-turbo'
                            Reminder: '/' should not be included in model_name

        Returns:
            int: The number of tokens in the text.

        Raises:
            ValueError: If the model_name is not a valid OpenAI model.
        """
        import tiktoken

        if "/" in model_name:
            raise ValueError("Non-OpenAI models are not supported")
        if model_name.lower().startswith("openai:"):
            model_name = model_name.lower().lstrip("openai:")
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(text)
        num_tokens = len(tokens)
        return num_tokens

    @staticmethod
    def compute_tokens_gemini(text: str, model_name: str) -> int:
        """
        Compute the number of tokens in a given text using Google Vertex AI.

        Args:
            text (str): The text to be tokenized.
            model_name (str): The name of the Gemini model. ex. 'gemini:gemini-1.5-flash'


        Returns:
            int: The number of tokens in the text.

        Raises:
            ValueError: If the model_id is not a valid Gemini model.
        """
        try:
            from akasha.utils.models.gemi import calculate_token

            if "/" in model_name:
                model_name = model_name.split("/")[-1]
            if model_name.lower().startswith("gemini:"):
                model_name = model_name.lower().lstrip("gemini:")
            # tokenizer = tokenization.get_tokenizer_for_model(model_name)
            # num_tokens = tokenizer.count_tokens(text).total_tokens
            num_tokens = calculate_token(text, model_name=model_name)
        except Exception:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(text))
            return num_tokens

        return num_tokens

    @classmethod
    def compute_tokens(
        cls,
        text: str,
        model_id: str,
        model_path: str = "./tokenizers",
        save_tokenizer: bool = True,
    ) -> int:
        """
        Compute the number of tokens in a given text using either huggingface or OpenAI tiktoken.

        If the model_id is an OpenAI model, the tiktoken library is used to tokenize the text.
        If the model_id is a Non-OpenAI model, the huggingface tokenizer is used to tokenize the text.

        Args:
            text (str): The text to be tokenized.
            model_id (str): The name of the model. ex. 'gpt-2', 'openai:gpt-3.5-turbo'
            model_path (str, optional): The path to the tokenizer. Defaults to './tokenizers'.
            save_tokenizer (bool, optional): Whether to save the tokenizer locally. Defaults to True.

        Returns:
            int: The number of tokens in the text.

        """
        model_type, model_name = separate_name(model_id)
        if model_type in ["openai", "gpt-3.5", "gpt"]:
            return cls.compute_tokens_openai(text, model_name)
        elif model_type in ["gemini", "google"]:
            return cls.compute_tokens_gemini(text, model_name)
        else:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(text))
            return num_tokens
