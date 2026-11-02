"""The agent with a scripted model: tools, memory and limits are ours to test; the model's wits are eval's."""

from datetime import date, timedelta
from decimal import Decimal

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from expense_tracker.agent import LIMIT_REACHED, Agent, Context, build_agent, reply, thread
from expense_tracker.expenses import Period, list_expenses
from expense_tracker.models import Expense
from tests.conftest import FixedRates, requires_db
from tests.scripted import Script, ScriptedModel, scripted

pytestmark: pytest.MarkDecorator = requires_db

ME: int = 111
CHAT: int = 5000
TODAY: date = date(2026, 11, 2)
WEEK: Period = Period(start=date(2026, 11, 1), end=date(2026, 11, 8))


def call(name: str, args: dict[str, object], id: str = "c1") -> AIMessage:
    tool_call: ToolCall = ToolCall(name=name, args=args, id=id)
    return AIMessage("", tool_calls=[tool_call])


def context(sessions: async_sessionmaker[AsyncSession]) -> Context:
    return Context(user_id=ME, sessions=sessions, rates=FixedRates(), today=TODAY)


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
    history: list[BaseMessage] = (
        await agent.aget_state({"configurable": {"thread_id": thread(CHAT, TODAY)}})
    ).values["messages"]
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
    assert answer == LIMIT_REACHED


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
            await restarted.aget_state({"configurable": {"thread_id": thread(CHAT, TODAY)}})
        ).values["messages"]
    assert [m.text for m in history] == ["привет", "Привет.", "помнишь?", "Помню."]


async def test_a_new_day_starts_a_new_history(sessions: async_sessionmaker[AsyncSession]) -> None:
    """Yesterday's chat is not re-sent to the model; yesterday's expenses are still in the database."""
    model: ScriptedModel = scripted(
        call("add_expenses", {"items": [{"amount": 60000, "category": "eating_out", "note": "фо"}]}),
        AIMessage("Записал."),
    )
    agent: Agent = build_agent(model, InMemorySaver())
    await reply(agent, CHAT, "фо 60к", context(sessions))
    tomorrow: Context = Context(
        user_id=ME, sessions=sessions, rates=FixedRates(), today=TODAY + timedelta(days=1)
    )
    model.messages = Script(
        [call("list_expenses", {"start": "2026-11-02", "end": "2026-11-02"}), AIMessage("Фо.")]
    )
    await reply(agent, CHAT, "что было вчера?", tomorrow)
    history: list[BaseMessage] = (
        await agent.aget_state({"configurable": {"thread_id": thread(CHAT, tomorrow.today)}})
    ).values["messages"]
    assert [m.text for m in history][0] == "что было вчера?"  # nothing of yesterday's turn
    assert "фо" in str(history[2].content)  # the answer came from the database


async def summary_answer(sessions: async_sessionmaker[AsyncSession], rates: FixedRates) -> str:
    """What the summary tool told the model about this week, next to the last one."""
    model: ScriptedModel = scripted(
        call(
            "add_expenses",
            {
                "items": [
                    {"amount": 100000, "category": "groceries"},
                    {"amount": "12.50", "currency": "USD", "category": "travel"},
                    {"amount": 50000, "category": "groceries", "spent_on": "2026-10-26"},
                ]
            },
        ),
        AIMessage("Записал."),
        call("summary", {"start": "2026-11-02", "end": "2026-11-08"}),
        AIMessage("Вот."),
    )
    agent: Agent = build_agent(model, InMemorySaver())
    ctx: Context = Context(user_id=ME, sessions=sessions, rates=rates, today=TODAY)
    await reply(agent, CHAT, "продукты 100к, билет 12.5$, продукты 26-го 50к", ctx)
    await reply(agent, CHAT, "сколько за неделю?", ctx)
    history: list[BaseMessage] = (
        await agent.aget_state({"configurable": {"thread_id": thread(CHAT, TODAY)}})
    ).values["messages"]
    return str(history[-2].content)


async def test_summary_totals_in_vnd_and_rub(sessions: async_sessionmaker[AsyncSession]) -> None:
    assert (await summary_answer(sessions, FixedRates())).splitlines() == [
        "travel USD: 12.50 (1), before 0",
        "groceries VND: 100 000 (1), before 50 000",
        "Total USD: 12.50, before 0",
        "Total VND: 100 000, before 50 000",
        "All in VND: 420 462, before 50 000",
        "All in RUB: 1 321, before 157",
        "Rate: Bank of Russia, 31.10.2026: 10 000 VND = 31.42 RUB",
    ]


async def test_summary_without_rates_keeps_the_totals(sessions: async_sessionmaker[AsyncSession]) -> None:
    lines: list[str] = (await summary_answer(sessions, FixedRates(None))).splitlines()
    assert lines[-3:] == [
        "Total USD: 12.50, before 0",
        "Total VND: 100 000, before 50 000",
        "No exchange rates now: no total in RUB.",
    ]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"amount": 1000000, "currency": "VND"}, "1 000 000 VND = 3 142.23 RUB (Bank of Russia, 31.10.2026)"),
        ({"amount": 1, "currency": "usd", "to": "vnd"}, "1 USD = 25 636.98 VND (Bank of Russia, 31.10.2026)"),
        ({"amount": 1, "currency": "XYZ"}, "The Bank of Russia has no rate for XYZ"),
    ],
)
async def test_convert(
    sessions: async_sessionmaker[AsyncSession], args: dict[str, object], expected: str
) -> None:
    agent: Agent = build_agent(scripted(call("convert", args), AIMessage("Вот.")), InMemorySaver())
    await reply(agent, CHAT, "сколько это в рублях?", context(sessions))
    history: list[BaseMessage] = (
        await agent.aget_state({"configurable": {"thread_id": thread(CHAT, TODAY)}})
    ).values["messages"]
    assert history[-2].content == expected
