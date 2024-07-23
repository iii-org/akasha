from langchain_openai import ChatOpenAI, AzureChatOpenAI
from outlines import models, generate
import os
from typing import Any, Union
from typing import Callable, Dict, List, Optional, Set, Tuple, Union


class MyOpenAI(ChatOpenAI):
    # format_model: Any = None

    # def choices_call(self, prompt, choices: list):

    #     if self.format_model is None:

    #         self.format_model = models.openai(
    #             self.model_name, api_key=os.environ["OPENAI_API_KEY"])

    #     generator = generate.choice(self.format_model, choices)

    #     return generator(prompt)

    def JSON_call(self, prompt, stop: Optional[List[str]] = None):

        return self.invoke(prompt,
                           response_format={"type": "json_object"},
                           stop=stop)


class MyAzureOpenAI(AzureChatOpenAI):
    # format_model: Any = None

    # def choices_call(self, prompt, choices: list):

    #     if self.format_model is None:

    #         self.format_model = models.azure_openai(
    #             self.deployment_name,
    #             api_key=os.environ["OPENAI_API_KEY"],
    #             azure_endpoint=os.environ["AZURE_API_BASE"],
    #             api_version=os.environ["AZURE_API_VERSION"])

    #     generator = generate.choice(self.format_model, choices)

    #     return generator(prompt)

    def JSON_call(self, prompt, stop: Optional[List[str]] = None):

        return self.invoke(prompt,
                           response_format={"type": "json_object"},
                           stop=stop)
