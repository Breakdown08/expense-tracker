"""The agent with a scripted model: tools, memory and limits are ours to test; the model's wits are eval's."""

from datetime import date, timedelta
from decimal import Decimal

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from expense_tracker.agent import (
    LIMIT_REACHED,
    Agent,
    Context,
    build_agent,
    reply,
    thread,
    without_past_thinking,
)
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
        )
    )
    answer: str = await reply(agent, CHAT, "фо 60к, кофе вчера 35к", context(sessions))
    rows: list[Expense] = await recorded(sessions)
    assert answer.splitlines() == [
        "Записал:",
        "60 000 ₫ Кафе «фо»",
        "35 000 ₫ Кафе «кофе»",
        "Всего в сообщении: 95 000 ₫",
        "За 1 ноября: 35 000 ₫",
        "За 2 ноября: 60 000 ₫",
    ]
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


async def test_ids_reach_the_model_and_not_the_chat(sessions: async_sessionmaker[AsyncSession]) -> None:
    agent: Agent = build_agent(
        scripted(call("add_expenses", {"items": [{"amount": 60000, "category": "eating_out"}]})),
        InMemorySaver(),
    )
    answer: str = await reply(agent, CHAT, "фо 60к", context(sessions))
    assert "#" not in answer
    history: list[BaseMessage] = (
        await agent.aget_state({"configurable": {"thread_id": thread(CHAT, TODAY)}})
    ).values["messages"]
    assert "Record ids, in the order above: #" in str(history[-1].content)


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


async def tool_answers(
    sessions: async_sessionmaker[AsyncSession], *steps: tuple[str, AIMessage]
) -> list[str]:
    """What the tools told the chat, one per step, in one chat."""
    script: list[AIMessage] = []
    step: tuple[str, AIMessage]
    for step in steps:
        script.append(step[1])
    agent: Agent = build_agent(scripted(*script), InMemorySaver())
    for step in steps:
        await reply(agent, CHAT, step[0], context(sessions))
    history: list[BaseMessage] = (
        await agent.aget_state({"configurable": {"thread_id": thread(CHAT, TODAY)}})
    ).values["messages"]
    return [m.artifact or str(m.content) for m in history if isinstance(m, ToolMessage)]


async def test_totals_after_a_record_come_from_the_code(sessions: async_sessionmaker[AsyncSession]) -> None:
    """The message's total and the day's: earlier today counts for the day, not for the message."""
    answers: list[str] = await tool_answers(
        sessions,
        ("фо 60к", call("add_expenses", {"items": [{"amount": 60000, "category": "eating_out"}]})),
        (
            "кофе 35к, билет 12.5$, вчера грэб 45к",
            call(
                "add_expenses",
                {
                    "items": [
                        {"amount": 35000, "category": "eating_out"},
                        {"amount": "12.50", "currency": "USD", "category": "travel"},
                        {"amount": 45000, "category": "transport", "spent_on": "2026-11-01"},
                    ]
                },
                id="c2",
            ),
        ),
    )
    assert answers[0].splitlines()[-1] == "За 2 ноября: 60 000 ₫"
    assert "Всего в сообщении" not in answers[0]  # one line is its own total
    assert answers[1].splitlines()[-3:] == [
        "Всего в сообщении: 12.50 USD, 80 000 ₫",
        "За 1 ноября: 45 000 ₫",
        "За 2 ноября: 12.50 USD, 95 000 ₫",
    ]


async def test_totals_after_a_correction_and_a_deletion(sessions: async_sessionmaker[AsyncSession]) -> None:
    add: AIMessage = call(
        "add_expenses",
        {"items": [{"amount": 60000, "category": "eating_out"}, {"amount": 35000, "category": "eating_out"}]},
    )
    first: list[str] = await tool_answers(sessions, ("фо 60к, кофе 35к", add))
    assert first[0].endswith("За 2 ноября: 95 000 ₫")
    rows: list[Expense] = await recorded(sessions)
    edit: AIMessage = call("edit_expense", {"expense_id": rows[0].id, "changes": {"amount": 50000}})
    assert (await tool_answers(sessions, ("нет, было 50", edit)))[0].endswith("За 2 ноября: 85 000 ₫")
    remove: AIMessage = call("delete_expense", {"expense_id": rows[1].id})
    assert (await tool_answers(sessions, ("кофе не было", remove)))[0].splitlines() == [
        "Удалил: 35 000 ₫ Кафе",
        "За 2 ноября: 50 000 ₫",
    ]
    remove = call("delete_expense", {"expense_id": rows[0].id})
    assert (await tool_answers(sessions, ("и фо", remove)))[0].endswith("За 2 ноября: 0")
    remove = call("delete_expense", {"expense_id": rows[0].id})
    assert (await tool_answers(sessions, ("ещё раз", remove)))[0] == "Такой записи нет"


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
        ({"amount": 1000000, "currency": "VND"}, "1 000 000 VND = 3 142.23 RUB (ЦБ, 31.10.2026)"),
        ({"amount": 1, "currency": "usd", "to": "vnd"}, "1 USD = 25 636.98 VND (ЦБ, 31.10.2026)"),
        ({"amount": 1, "currency": "XYZ"}, "У ЦБ нет курса XYZ"),
    ],
)
async def test_convert(
    sessions: async_sessionmaker[AsyncSession], args: dict[str, object], expected: str
) -> None:
    agent: Agent = build_agent(scripted(call("convert", args)), InMemorySaver())
    assert await reply(agent, CHAT, "сколько это в рублях?", context(sessions)) == expected


def add(amount: int, id: str = "c1") -> AIMessage:
    return call("add_expenses", {"items": [{"amount": amount, "category": "groceries"}]}, id=id)


async def test_a_direct_answer_spends_no_second_model_call(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    script: Script = Script([add(60000), AIMessage("never reached")])
    agent: Agent = build_agent(ScriptedModel(messages=script), InMemorySaver())
    answer: str = await reply(agent, CHAT, "продукты 60к", context(sessions))
    assert answer.startswith("Записал:")
    assert script.answers == [AIMessage("never reached")]


async def test_a_question_about_a_day_gets_the_report_as_is(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    agent: Agent = build_agent(
        scripted(add(60000), call("report", {"start": "2026-11-02", "end": "2026-11-02"})), InMemorySaver()
    )
    await reply(agent, CHAT, "продукты 60к", context(sessions))
    answer: str = await reply(agent, CHAT, "сколько сегодня?", context(sessions))
    assert answer.splitlines()[0] == "Сегодня, 2 ноября"
    assert answer.splitlines()[-2:] == [
        "Итого: 60 000 ₫ · 189 ₽",
        "Курс ЦБ на 31.10.2026: 10 000 ₫ = 31.42 ₽",
    ]


async def test_several_direct_answers_all_reach_the_user(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    both: AIMessage = AIMessage(
        "",
        tool_calls=[
            ToolCall(
                name="add_expenses",
                args={"items": [{"amount": 60000, "category": "groceries"}]},
                id="c1",
            ),
            ToolCall(name="convert", args={"amount": 1, "currency": "USD"}, id="c2"),
        ],
    )
    answer: str = await reply(build_agent(scripted(both)), CHAT, "продукты 60к и доллар?", context(sessions))
    assert answer.startswith("Записал:")
    assert "1 USD = " in answer


async def test_analysis_goes_back_to_the_model(sessions: async_sessionmaker[AsyncSession]) -> None:
    agent: Agent = build_agent(
        scripted(
            call("summary", {"start": "2026-11-02", "end": "2026-11-08"}), AIMessage("Меньше, чем было.")
        )
    )
    assert await reply(agent, CHAT, "почему меньше?", context(sessions)) == "Меньше, чем было."


def test_past_turns_lose_thinking_and_the_current_one_keeps_it() -> None:
    """A changed prompt invalidates thinking in today's history; a tool loop still needs its own."""
    idea: dict[str, str] = {"type": "thinking", "thinking": "…", "signature": "old"}
    said: dict[str, str] = {"type": "text", "text": "Записал."}
    now: AIMessage = AIMessage(content=[{**idea, "signature": "new"}, said])
    messages: list[AnyMessage] = [
        HumanMessage("фо 60к"),
        AIMessage(content=[idea, said]),
        HumanMessage("кофе 35к"),
        now,
        ToolMessage("Записал", tool_call_id="c1"),
    ]
    sent: list[AnyMessage] = without_past_thinking(messages)
    assert sent[1].content == [said]
    assert sent[3] is now
    assert messages[1].content == [idea, said]  # the checkpoint's own history stays whole
