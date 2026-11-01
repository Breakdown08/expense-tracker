"""A chat model that answers from a script: the agent's plumbing is tested without a provider."""

from collections.abc import Iterator, Sequence

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable


class ScriptedModel(GenericFakeChatModel):
    """Answers with the given messages in order; tools are accepted and ignored."""

    def bind_tools(
        self, tools: Sequence[object], *, tool_choice: str | None = None, **kwargs: object
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


class Script(Iterator[AIMessage]):
    """A queue of answers; an exception in it is raised once, and the queue goes on."""

    def __init__(self, answers: Sequence[AIMessage | Exception]) -> None:
        self.answers: list[AIMessage | Exception] = list(answers)

    def __next__(self) -> AIMessage:
        if not self.answers:
            raise StopIteration
        answer: AIMessage | Exception = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def scripted(*answers: AIMessage | Exception) -> ScriptedModel:
    return ScriptedModel(messages=Script(answers))
