"""Reusable test doubles shared across feature-owned tests."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatModel(BaseChatModel):
    """Small chat-model double for event and Agent contract tests."""

    chunks: list

    @property
    def _llm_type(self):
        return "fake-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    def stream(self, _messages):
        yield from self.chunks

    def get_num_tokens(self, text):
        return len(text)
