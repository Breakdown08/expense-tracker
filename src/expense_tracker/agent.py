"""Expense agent: LangChain `create_agent` with tools over `expenses.py`. It knows nothing of posts."""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

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
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    dynamic_prompt,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from expense_tracker import expenses
from expense_tracker.expenses import Category, CategoryTotal, ExpenseChanges, NewExpense, Period
from expense_tracker.models import Expense


@dataclass(frozen=True)
class Context:
    """Per message: who writes, where the database is, and what day it is for them."""

    user_id: int
    sessions: async_sessionmaker[AsyncSession]
    today: date


class ExpenseInput(BaseModel):
    amount: Decimal = Field(gt=0, description="Full amount: «60к» is 60000, «1,5 млн» is 1500000")
    currency: str = Field(default="VND", pattern=r"^[A-Z]{3}$")
    category: Category
    note: str | None = Field(default=None, description="What it was, in the user's words")
    spent_on: date | None = Field(default=None, description="Omit for today")


def line(row: Expense) -> str:
    note: str = f" «{row.note}»" if row.note else ""
    return f"#{row.id} {row.spent_on} {row.amount:,.0f} {row.currency} {row.category}{note}".replace(",", " ")


def period(start: date, end: date) -> Period:
    """The model names both days inclusively, as people do; Period ends the day after."""
    return Period(start=start, end=end + timedelta(days=1))


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
    return "Recorded:\n" + "\n".join(line(r) for r in rows)


@tool
async def edit_expense(expense_id: int, changes: ExpenseChanges, runtime: ToolRuntime[Context]) -> str:
    """Correct a recorded expense by its #id; only the given fields change."""
    session: AsyncSession
    async with runtime.context.sessions() as session:
        row: Expense | None = await expenses.edit_expense(session, expense_id, changes)
    return f"Updated: {line(row)}" if row else f"No expense #{expense_id}"


@tool
async def delete_expense(expense_id: int, runtime: ToolRuntime[Context]) -> str:
    """Delete a recorded expense by its #id."""
    session: AsyncSession
    async with runtime.context.sessions() as session:
        deleted: bool = await expenses.delete_expense(session, expense_id)
    return f"Deleted #{expense_id}" if deleted else f"No expense #{expense_id}"


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
async def summary(start: date, end: date, runtime: ToolRuntime[Context]) -> str:
    """Totals per category from start to end (both included), next to the same number of days before."""
    session: AsyncSession
    async with runtime.context.sessions() as session:
        totals: list[CategoryTotal] = await expenses.summary(session, period(start, end))
    return (
        "\n".join(
            f"{t.category} {t.currency}: {t.total:,.0f} ({t.count}), before {t.previous_total:,.0f}".replace(
                ",", " "
            )
            for t in totals
        )
        or "No expenses"
    )


TOOLS: list[BaseTool] = [add_expenses, edit_expense, delete_expense, list_expenses, summary]

PROMPT: str = """You keep a family's expense book in Telegram. The family lives in Vietnam and pays in VND \
unless they name another currency. Today is {today}.

- Messages are short: «фо 60к», «grab 45k, кофе 35к». «к»/«k» means thousand, «млн»/«tr» million.
- Record everything from one message with a single add_expenses call.
- «нет, было 50» corrects the expense just recorded: use its #id from the tool result.
- Never compute totals yourself: use summary or list_expenses and retell their numbers.
- If the amount or what it was is unclear, ask instead of guessing.
- Reply briefly in the user's language. Amounts with spaces between thousands: 60 000 VND."""


@dynamic_prompt
def system_prompt(request: ModelRequest[Context]) -> str:
    return PROMPT.format(today=request.runtime.context.today.isoformat())


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

# As create_agent declares its result. ty 0.0.95 does not see that a TypedDict class has
# __optional_keys__ and rejects the bound; mypy and the runtime agree it is there.
type Agent = CompiledStateGraph[AgentState[None], Context, InputAgentState, OutputAgentState[None]]  # ty: ignore[invalid-type-arguments]


def build_agent(model: BaseChatModel, checkpointer: Checkpointer = None, retry_delay: float = 5.0) -> Agent:
    # Any: AgentMiddleware is invariant in its state, so LangChain's own limit middlewares (each with
    # its own state type) do not fit create_agent's list under mypy without widening it here.
    chain: list[AgentMiddleware[Any, Context]] = [
        system_prompt,
        # Old turns fold into a summary: every model call re-sends the whole history.
        SummarizationMiddleware(model, trigger=("tokens", 3000), keep=("messages", 12)),
        ModelRetryMiddleware(retry_on=TRANSIENT, max_retries=3, initial_delay=retry_delay, max_delay=30.0),
        ModelCallLimitMiddleware(run_limit=6),  # one message never loops for long
        ToolCallLimitMiddleware(run_limit=8),
    ]
    return create_agent(model, TOOLS, middleware=chain, context_schema=Context, checkpointer=checkpointer)


async def reply(agent: Agent, chat_id: int, text: str, context: Context) -> str:
    """One user message in, the agent's last answer out. The chat is the memory thread."""
    result: dict[str, object] = await agent.ainvoke(
        {"messages": [HumanMessage(text)]},
        config={"configurable": {"thread_id": str(chat_id)}},
        context=context,
    )
    messages: object = result["messages"]
    assert isinstance(messages, list)
    last: object = messages[-1]
    assert isinstance(last, AIMessage)
    return last.text  # text blocks only: gpt-oss reasoning stays out of Telegram
