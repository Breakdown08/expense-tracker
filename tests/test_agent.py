"""The agent with a scripted model: tools, memory and limits are ours to test; the model's wits are eval's."""

from collections.abc import Iterator, Sequence
from datetime import date
from decimal import Decimal

import groq
import httpx
import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from expense_tracker.agent import Agent, Context, build_agent, reply
from expense_tracker.expenses import Period, list_expenses
from expense_tracker.models import Expense
from tests.conftest import requires_db

pytestmark: pytest.MarkDecorator = requires_db

ME: int = 111
CHAT: int = 5000
TODAY: date = date(2026, 11, 2)
WEEK: Period = Period(start=date(2026, 11, 1), end=date(2026, 11, 8))


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


def call(name: str, args: dict[str, object], id: str = "c1") -> AIMessage:
    tool_call: ToolCall = ToolCall(name=name, args=args, id=id)
    return AIMessage("", tool_calls=[tool_call])


def context(sessions: async_sessionmaker[AsyncSession]) -> Context:
    return Context(user_id=ME, sessions=sessions, today=TODAY)


async def recorded(sessions: async_sessionmaker[AsyncSession]) -> list[Expense]:
    session: AsyncSession
    async with sessions() as session:
        return await list_expenses(session, WEEK)


async def test_one_message_several_expenses(sessions: async_sessionmaker[AsyncSession]) -> None:
    agent: Agent = build_agent(
        scripted(
            call(
                "add_expenses",
                {
                    "items": [
                        {"amount": 60000, "category": "eating_out", "note": "фо"},
                        {
                            "amount": "35000",
                            "category": "eating_out",
                            "note": "кофе",
                            "spent_on": "2026-11-01",
                        },
                    ]
                },
            ),
            AIMessage("Записал: фо 60 000, кофе 35 000."),
        )
    )
    answer: str = await reply(agent, CHAT, "фо 60к, кофе вчера 35к", context(sessions))
    assert answer == "Записал: фо 60 000, кофе 35 000."
    rows: list[Expense] = await recorded(sessions)
    assert [(r.note, r.amount, r.currency, r.spent_on, r.user_id) for r in rows] == [
        ("кофе", Decimal(35000), "VND", date(2026, 11, 1), ME),
        ("фо", Decimal(60000), "VND", TODAY, ME),  # no date from the model: today of the context
    ]


async def test_correction_uses_memory_of_the_chat(sessions: async_sessionmaker[AsyncSession]) -> None:
    """«нет, было 50» needs the #id from the previous turn: the checkpointer keeps it."""
    model: ScriptedModel = scripted(
        call("add_expenses", {"items": [{"amount": 60000, "category": "eating_out", "note": "фо"}]}),
        AIMessage("Записал."),
    )
    agent: Agent = build_agent(model, InMemorySaver())
    await reply(agent, CHAT, "фо 60к", context(sessions))
    row: Expense = (await recorded(sessions))[0]
    model.messages = Script(
        [
            call("edit_expense", {"expense_id": row.id, "changes": {"amount": 50000}}, id="c2"),
            AIMessage("Исправил."),
        ]
    )
    await reply(agent, CHAT, "нет, было 50", context(sessions))
    assert [r.amount for r in await recorded(sessions)] == [Decimal(50000)]
    history: list[BaseMessage] = (await agent.aget_state({"configurable": {"thread_id": str(CHAT)}})).values[
        "messages"
    ]
    assert f"#{row.id}" in str(history[2].content)  # the tool result the model read the id from


async def test_bad_tool_arguments_go_back_to_the_model(sessions: async_sessionmaker[AsyncSession]) -> None:
    agent: Agent = build_agent(
        scripted(
            call("add_expenses", {"items": [{"amount": -5, "category": "eating_out"}]}),
            AIMessage("Сумма должна быть положительной — сколько было?"),
        )
    )
    answer: str = await reply(agent, CHAT, "фо минус 5", context(sessions))
    assert answer.startswith("Сумма должна")
    assert await recorded(sessions) == []


async def test_endless_tool_calls_stop(sessions: async_sessionmaker[AsyncSession]) -> None:
    calls: list[AIMessage] = [
        call("list_expenses", {"start": "2026-11-01", "end": "2026-11-07"}, f"c{i}") for i in range(20)
    ]
    agent: Agent = build_agent(scripted(*calls))
    answer: str = await reply(agent, CHAT, "покажи траты", context(sessions))
    assert "limit" in answer.lower()


def rate_limited() -> groq.RateLimitError:
    response: httpx.Response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
    return groq.RateLimitError("Rate limit reached", response=response, body=None)


def unauthorized() -> groq.AuthenticationError:
    response: httpx.Response = httpx.Response(401, request=httpx.Request("POST", "https://api.groq.com"))
    return groq.AuthenticationError("Invalid API Key", response=response, body=None)


async def test_rate_limit_is_retried(sessions: async_sessionmaker[AsyncSession]) -> None:
    agent: Agent = build_agent(scripted(rate_limited(), AIMessage("Готово.")), retry_delay=0.01)
    assert await reply(agent, CHAT, "сколько за неделю?", context(sessions)) == "Готово."


async def test_wrong_key_is_not_retried(sessions: async_sessionmaker[AsyncSession]) -> None:
    """A retry would get the same 401: it surfaces at once, and the bot tells the user."""
    script: Script = Script([unauthorized(), AIMessage("never reached")])
    agent: Agent = build_agent(ScriptedModel(messages=script), retry_delay=0.01)
    with pytest.raises(groq.AuthenticationError):
        await reply(agent, CHAT, "сколько за неделю?", context(sessions))
    assert len(script.answers) == 1  # the model was called once


async def test_memory_survives_restart(test_db_url: str, sessions: async_sessionmaker[AsyncSession]) -> None:
    """Postgres checkpointer: a new agent (the bot restarted) continues the same chat."""
    saver: AsyncPostgresSaver
    async with AsyncPostgresSaver.from_conn_string(test_db_url) as saver:
        await saver.setup()
        await reply(build_agent(scripted(AIMessage("Привет.")), saver), CHAT, "привет", context(sessions))
    async with AsyncPostgresSaver.from_conn_string(test_db_url) as saver:
        restarted: Agent = build_agent(scripted(AIMessage("Помню.")), saver)
        await reply(restarted, CHAT, "помнишь?", context(sessions))
        history: list[BaseMessage] = (
            await restarted.aget_state({"configurable": {"thread_id": str(CHAT)}})
        ).values["messages"]
    assert [m.text for m in history] == ["привет", "Привет.", "помнишь?", "Помню."]
