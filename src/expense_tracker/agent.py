"""Expense agent: LangChain `create_agent` with tools over `expenses.py`. It knows nothing of posts."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, override

import anthropic
import groq
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    InputAgentState,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelRetryMiddleware,
    OutputAgentState,
    ToolCallLimitMiddleware,
    dynamic_prompt,
    hook_config,
)
from langchain.agents.middleware.model_call_limit import ModelCallLimitState
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from expense_tracker import expenses, reports
from expense_tracker.expenses import Category, CategoryTotal, ExpenseChanges, NewExpense
from expense_tracker.models import Expense
from expense_tracker.rates import Rates, RateSource, UnknownCurrency
from expense_tracker.reports import ONE, entry, latest_rates, money, period, prices

log: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Context:
    """Per message: who writes, where the database and the rates are, and what day it is for them."""

    user_id: int
    sessions: async_sessionmaker[AsyncSession]
    rates: RateSource
    today: date


class ExpenseInput(BaseModel):
    amount: Decimal = Field(gt=0, description="Full amount: «60к» is 60000, «1,5 млн» is 1500000")
    currency: str = Field(default="VND", pattern=r"^[A-Z]{3}$")
    category: Category
    note: str | None = Field(default=None, description="What it was, in the user's words")
    spent_on: date | None = Field(default=None, description="Omit for today")


def line(row: Expense) -> str:
    note: str = f" «{row.note}»" if row.note else ""
    return f"#{row.id} {row.spent_on} {money(row.amount)} {row.currency} {row.category}{note}"


async def day_line(session: AsyncSession, day: date) -> str:
    return f"За {reports.day_name(day)}: {prices(await expenses.day_totals(session, day))}"


def grand_total(totals: list[CategoryTotal], rates: Rates | None) -> list[str]:
    """Every currency's total, then all of them in VND and RUB at the latest rate — counted here, not by
    the model. Without rates the per-currency totals still stand."""
    now: dict[str, Decimal] = {}
    before: dict[str, Decimal] = {}
    t: CategoryTotal
    for t in totals:
        now[t.currency] = now.get(t.currency, Decimal(0)) + t.total
        before[t.currency] = before.get(t.currency, Decimal(0)) + t.previous_total
    lines: list[str] = [
        f"Total {currency}: {money(now[currency])}, before {money(before[currency])}" for currency in now
    ]
    if rates is None:
        return [*lines, "No exchange rates now: no total in RUB."]
    unknown: list[str] = [currency for currency in now if currency not in rates.rub]
    known: list[str] = [currency for currency in now if currency in rates.rub]
    target: str
    for target in ("VND", "RUB"):
        sum_now: Decimal = sum((rates.convert(now[c], c, target) for c in known), Decimal(0))
        sum_before: Decimal = sum((rates.convert(before[c], c, target) for c in known), Decimal(0))
        whole: tuple[str, str] = (money(sum_now.quantize(ONE)), money(sum_before.quantize(ONE)))
        lines.append(f"All in {target}: {whole[0]}, before {whole[1]}")
    vnd: Decimal = rates.convert(Decimal(10000), "VND", "RUB")
    lines.append(f"Rate: Bank of Russia, {rates.on:%d.%m.%Y}: 10 000 VND = {vnd:.2f} RUB")
    if unknown:
        lines.append(f"Left out of «All in», no rate: {', '.join(unknown)}")
    return lines


@tool
async def add_expenses(items: list[ExpenseInput], runtime: ToolRuntime[Context]) -> str:
    """Record one or more expenses from a single message, all in one call."""
    ctx: Context = runtime.context
    new: list[NewExpense] = [
        NewExpense(**item.model_dump(exclude={"spent_on"}), spent_on=item.spent_on or ctx.today)
        for item in items
    ]
    session: AsyncSession
    async with ctx.sessions() as session:
        rows: list[Expense] = await expenses.add_expenses(session, ctx.user_id, new)
        days: list[str] = [await day_line(session, day) for day in sorted({r.spent_on for r in rows})]
    recorded: list[str] = [entry(r) for r in rows]
    if len(rows) > 1:
        message: dict[str, Decimal] = {}
        row: Expense
        for row in rows:
            message[row.currency] = message.get(row.currency, Decimal(0)) + row.amount
        recorded.append(f"Всего в сообщении: {prices(message)}")
    return "\n".join(["Записал:", *recorded, *days])


@tool
async def edit_expense(expense_id: int, changes: ExpenseChanges, runtime: ToolRuntime[Context]) -> str:
    """Correct a recorded expense by its #id; only the given fields change."""
    session: AsyncSession
    async with runtime.context.sessions() as session:
        row: Expense | None = await expenses.edit_expense(session, expense_id, changes)
        if row is None:
            return f"Нет записи #{expense_id}"
        return f"Исправил: {entry(row)}\n{await day_line(session, row.spent_on)}"


@tool
async def delete_expense(expense_id: int, runtime: ToolRuntime[Context]) -> str:
    """Delete a recorded expense by its #id."""
    session: AsyncSession
    async with runtime.context.sessions() as session:
        day: date | None = await expenses.delete_expense(session, expense_id)
        if day is None:
            return f"Нет записи #{expense_id}"
        return f"Удалил #{expense_id}\n{await day_line(session, day)}"


@tool
async def list_expenses(
    start: date, end: date, runtime: ToolRuntime[Context], category: Category | None = None
) -> str:
    """Expenses from start to end, both days included, optionally of one category."""
    session: AsyncSession
    async with runtime.context.sessions() as session:
        rows: list[Expense] = await expenses.list_expenses(session, period(start, end), category)
    return "\n".join(line(r) for r in rows) or "No expenses"


@tool
async def report(start: date, end: date, runtime: ToolRuntime[Context]) -> str:
    """What was spent from start to end (both included), with the total in VND and RUB. Reaches the user
    word for word."""
    ctx: Context = runtime.context
    return await reports.report(ctx.sessions, ctx.rates, ctx.today, start, end)


@tool
async def summary(start: date, end: date, runtime: ToolRuntime[Context]) -> str:
    """Totals per category from start to end (both included), next to the same number of days before.
    For you to analyse; for a plain total use report."""
    ctx: Context = runtime.context
    session: AsyncSession
    async with ctx.sessions() as session:
        totals: list[CategoryTotal] = await expenses.summary(session, period(start, end))
    if not totals:
        return "No expenses"
    rates: Rates | None = await latest_rates(ctx.rates, ctx.today)
    categories: list[str] = [
        f"{t.category} {t.currency}: {money(t.total)} ({t.count}), before {money(t.previous_total)}"
        for t in totals
    ]
    return "\n".join([*categories, *grand_total(totals, rates)])


@tool
async def convert(amount: Decimal, currency: str, runtime: ToolRuntime[Context], to: str = "RUB") -> str:
    """Convert an amount between currencies (VND, RUB, USD…) at the Bank of Russia's latest rate."""
    ctx: Context = runtime.context
    try:
        rates: Rates = await ctx.rates.latest(ctx.today)
        result: Decimal = rates.convert(amount, currency.upper(), to.upper())
    except UnknownCurrency as error:
        return f"У ЦБ нет курса {error}"
    except Exception:
        log.warning("no exchange rates for convert", exc_info=True)
        return "Курсы сейчас недоступны"
    return (
        f"{money(amount)} {currency.upper()} = {money(result.quantize(Decimal('0.01')))} {to.upper()} "
        f"(ЦБ, {rates.on:%d.%m.%Y})"
    )


TOOLS: list[BaseTool] = [
    add_expenses,
    edit_expense,
    delete_expense,
    list_expenses,
    report,
    summary,
    convert,
]

# Their result is the answer: no second model call to retell it.
DIRECT: frozenset[str] = frozenset({"add_expenses", "edit_expense", "delete_expense", "report", "convert"})

PROMPT: str = """You keep a family's expense book in Telegram. The family lives in Vietnam and pays in VND \
unless they name another currency. Today is {today:%A, %Y-%m-%d}.

- Two people share one chat and one wallet. Each message starts with its author: «[Кирилл] фо 60к». \
Don't repeat the tag in replies; anyone may correct anyone's expense.
- Messages are short: «фо 60к», «grab 45k, кофе 35к». «к»/«k» means thousand, «млн»/«tr» million.
- Record everything from one message with a single add_expenses call.
- «нет, было 50» corrects the expense just recorded: use its #id from the tool result.
- The chat history holds only today's messages. Anything earlier — yesterday, last month — \
comes from list_expenses or summary, never from memory.
- The results of add_expenses, edit_expense, delete_expense, report and convert go to the user as they \
are and end your turn, so when one message asks for several things make all the calls at once.
- «Сколько потратили», «что записано» for a day or a period is report: the user gets it word for word. \
«Неделя» without a date means from this week's Monday to today. \
Use summary or list_expenses only when the answer takes more than a total: comparing periods, one category, \
what grew, the total in another currency than VND and RUB. Then never compute totals yourself. \
Past expenses convert at the latest rate, not at the rate of their day. \
For any other conversion use convert; never convert currencies yourself.
- Categories: groceries — food and drinks from shops and markets; eating_out — cafés, restaurants, \
coffee, delivery; transport — Grab, taxi, fuel, parking; housing — rent, utilities; \
subscriptions — internet, phone plan, streaming, apps; \
household — things for the home, cleaning, repairs; health — doctors, pharmacy; clothes; \
leisure — entertainment, hobbies, gifts; travel — trips, tickets, hotels; other — nothing above fits.
- If the amount or what it was is unclear, ask instead of guessing.
- Reply briefly in the user's language. Amounts with spaces between thousands: 60 000 VND."""


@dynamic_prompt
def system_prompt(request: ModelRequest[Context]) -> str:
    return PROMPT.format(today=request.runtime.context.today)


# Worth retrying: rate limits (Groq free tier: 8K tokens a minute), network, provider 5xx.
# Not worth it: a wrong key or a malformed request fails the same way every time.
TRANSIENT: tuple[type[Exception], ...] = (
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.InternalServerError,
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)

LIMIT_REACHED: str = (
    "Не справился за разумное число шагов и остановился. "
    "Что успело записаться — спроси «что записано сегодня»."
)


def tool_results(messages: Sequence[BaseMessage]) -> list[ToolMessage]:
    """The tool messages that close the history, in order."""
    results: list[ToolMessage] = []
    message: BaseMessage
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        results.insert(0, message)
    return results


class CallLimit(ModelCallLimitMiddleware[Any, Context]):
    """LangChain's limit, with the reason in the family's language instead of «Model call limits exceeded»."""

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self, state: ModelCallLimitState[Any], runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        update: dict[str, Any] | None = super().before_model(state, runtime)
        if update is not None:
            update["messages"] = [AIMessage(LIMIT_REACHED)]
        return update


class DirectAnswer(AgentMiddleware[Any, Context]):
    """Ends the turn before the model is called again when the tools have already answered.

    Not `return_direct`: that also ends the turn on a tool error, and a wrong argument has to go back
    to the model to be fixed, not to the chat.
    """

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(self, state: AgentState[Any], runtime: Runtime[Context]) -> dict[str, Any] | None:
        results: list[ToolMessage] = tool_results(state["messages"])
        if results and all(m.name in DIRECT and m.status != "error" for m in results):
            return {"jump_to": "end"}
        return None


# As create_agent declares its result. ty 0.0.95 does not see that a TypedDict class has
# __optional_keys__ and rejects the bound; mypy and the runtime agree it is there.
type Agent = CompiledStateGraph[AgentState[None], Context, InputAgentState, OutputAgentState[None]]  # ty: ignore[invalid-type-arguments]


def build_agent(model: BaseChatModel, checkpointer: Checkpointer = None, retry_delay: float = 5.0) -> Agent:
    # Any: AgentMiddleware is invariant in its state, so LangChain's own limit middlewares (each with
    # its own state type) do not fit create_agent's list under mypy without widening it here.
    chain: list[AgentMiddleware[Any, Context]] = [
        DirectAnswer(),
        system_prompt,
        ModelRetryMiddleware(retry_on=TRANSIENT, max_retries=3, initial_delay=retry_delay, max_delay=30.0),
        CallLimit(run_limit=6),  # one message never loops for long
        ToolCallLimitMiddleware(run_limit=8),
    ]
    return create_agent(model, TOOLS, middleware=chain, context_schema=Context, checkpointer=checkpointer)


def thread(chat_id: int, day: date) -> str:
    """A chat's memory lasts one day: corrections refer to what was just written, and earlier
    expenses are in the database. A year of use costs no more tokens than its first day."""
    return f"{chat_id}:{day.isoformat()}"


async def reply(agent: Agent, chat_id: int, text: str, context: Context) -> str:
    """One user message in, the agent's last answer out, in today's thread of the chat."""
    result: dict[str, object] = await agent.ainvoke(
        {"messages": [HumanMessage(text)]},
        config={"configurable": {"thread_id": thread(chat_id, context.today)}},
        context=context,
    )
    messages: object = result["messages"]
    assert isinstance(messages, list)
    direct: list[ToolMessage] = tool_results(messages)
    if direct:
        return "\n\n".join(m.text for m in direct)
    last: object = messages[-1]
    assert isinstance(last, AIMessage)
    return last.text  # text blocks only: gpt-oss reasoning stays out of Telegram
