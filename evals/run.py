"""How well a real model turns phrases into expenses and answers about them: `uv run python -m evals.run`.

Costs money or free-tier quota, so it runs by hand, not in CI. The model is LLM_MODEL from `.env`;
DATABASE_URL must be reachable from here (locally: localhost, not `db`). Expenses land in a
`<name>_eval` database next to it, dropped afterwards. The tests check the tools; this checks the model.
"""

import asyncio
import json
import re
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

import psycopg
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from psycopg import sql
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from expense_tracker.agent import Agent, Context, build_agent, remember, reply, thread
from expense_tracker.config import Settings
from expense_tracker.db import make_engine, make_sessions, upgrade
from expense_tracker.expenses import Category, NewExpense, add_expenses
from expense_tracker.models import Expense
from expense_tracker.rates import Rates
from expense_tracker.reports import Report, report

HERE: Path = Path(__file__).parent
CHAT: int = 1
USER: int = 1
TODAY: date = date(2026, 11, 2)  # a Monday: «в субботу» is the 31st of October

type Key = tuple[Decimal, str, date]


class Spent(BaseModel):
    amount: Decimal
    currency: str = "VND"
    category: Category
    days_ago: int = 0
    note: str | None = None

    def key(self) -> Key:
        return (self.amount, self.currency, TODAY - timedelta(days=self.days_ago))


class Case(BaseModel):
    messages: list[str]  # sent in turn, in one chat: corrections need the turn before
    expected: list[Spent] = []  # what the last message leaves added to the book
    book: bool = False  # start from book.jsonl instead of an empty book
    gone: list[int] = []  # places in book.jsonl the last message leaves deleted
    shown: int | None = None  # days ago of a day list a command showed before the messages
    answer: list[str] = []  # patterns the last answer must match, all of them


class EvalRates:
    """Round rates: the totals in `questions.jsonl` are worked out from them by hand."""

    async def latest(self, today: date) -> Rates:
        return Rates(on=today, rub={"RUB": Decimal(1), "VND": Decimal("0.0032"), "USD": Decimal(80)})


class Verdict(BaseModel):
    amounts: bool  # amount, currency and day are right
    exact: bool  # and the category too


def judge(expected: list[Spent], got: list[Expense]) -> Verdict:
    amounts: bool = Counter(s.key() for s in expected) == Counter(
        (r.amount, r.currency, r.spent_on) for r in got
    )
    exact: bool = Counter((*s.key(), s.category.value) for s in expected) == Counter(
        (r.amount, r.currency, r.spent_on, r.category) for r in got
    )
    return Verdict(amounts=amounts, exact=exact)


def answered(case: Case, answer: str) -> bool:
    """Every number the case asks for is there, and no #id got out to the chat."""
    return all(re.search(p, answer) for p in case.answer) and not re.search(r"#\d", answer)


def final(case: Case, book: list[Spent]) -> list[Spent]:
    """What the book must hold after the last message."""
    start: list[Spent] = book if case.book else []
    return [s for i, s in enumerate(start) if i not in case.gone] + case.expected


def load(path: Path) -> list[Case]:
    return [Case.model_validate_json(row) for row in path.read_text().splitlines() if row]


def tokens(history: list[BaseMessage]) -> tuple[int, int]:
    used: list[AIMessage] = [m for m in history if isinstance(m, AIMessage)]
    return (
        sum(m.usage_metadata["input_tokens"] for m in used if m.usage_metadata),
        sum(m.usage_metadata["output_tokens"] for m in used if m.usage_metadata),
    )


async def run_case(
    case: Case, book: list[Spent], model: BaseChatModel, sessions: async_sessionmaker[AsyncSession]
) -> dict[str, object]:
    session: AsyncSession
    async with sessions() as session:
        await session.execute(delete(Expense))
        await session.commit()
        if case.book:
            await add_expenses(
                session,
                USER,
                [
                    NewExpense(
                        amount=s.amount,
                        currency=s.currency,
                        category=s.category,
                        note=s.note,
                        spent_on=TODAY - timedelta(days=s.days_ago),
                    )
                    for s in book
                ],
            )
    agent: Agent = build_agent(model, InMemorySaver())
    context: Context = Context(user_id=USER, sessions=sessions, rates=EvalRates(), today=TODAY)
    if case.shown is not None:
        day: date = TODAY - timedelta(days=case.shown)
        shown: Report = await report(sessions, EvalRates(), TODAY, day, day)
        await remember(agent, CHAT, TODAY, "[Кирилл] /день", shown.for_model())
    answer: str = ""
    message: str
    for message in case.messages:
        answer = await reply(agent, CHAT, f"[Кирилл] {message}", context)
    async with sessions() as session:
        got: list[Expense] = list((await session.scalars(select(Expense).order_by(Expense.id))).all())
    history: list[BaseMessage] = (
        await agent.aget_state({"configurable": {"thread_id": thread(CHAT, TODAY)}})
    ).values["messages"]
    verdict: Verdict = judge(final(case, book), got)
    used: tuple[int, int] = tokens(history)
    return {
        "messages": case.messages,
        **verdict.model_dump(),
        "answered": answered(case, answer),
        "got": [
            {
                "amount": str(r.amount),
                "currency": r.currency,
                "category": r.category,
                "spent_on": r.spent_on.isoformat(),
            }
            for r in got
        ],
        "answer": answer,
        "input_tokens": used[0],
        "output_tokens": used[1],
    }


async def evaluate(settings: Settings, url: str) -> list[dict[str, object]]:
    model: BaseChatModel = init_chat_model(
        settings.llm_model, api_key=settings.llm_api_key.get_secret_value()
    )
    engine: AsyncEngine = make_engine(url)
    sessions: async_sessionmaker[AsyncSession] = make_sessions(engine)
    results: list[dict[str, object]] = []
    book: list[Spent] = [Spent.model_validate_json(r) for r in (HERE / "book.jsonl").read_text().splitlines()]
    case: Case
    try:
        for case in load(HERE / "phrases.jsonl") + load(HERE / "questions.jsonl"):
            result: dict[str, object] = await run_case(case, book, model, sessions)
            results.append(result)
            if not (result["exact"] and result["answered"]):
                print(f"miss: {case.messages} → {result['got']}\n{result['answer']}")
    finally:
        await engine.dispose()
    return results


def main() -> None:
    settings: Settings = Settings()
    parts: SplitResult = urlsplit(settings.database_url)
    name: str = f"{parts.path.lstrip('/')}_eval"
    url: str = parts._replace(path=f"/{name}").geturl()
    admin: psycopg.Connection[tuple[object, ...]]
    with psycopg.connect(settings.database_url, autocommit=True) as admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        upgrade(url)
        results: list[dict[str, object]] = asyncio.run(evaluate(settings, url))
    finally:
        with psycopg.connect(settings.database_url, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
    slug: str = settings.llm_model.split(":", 1)[1].replace("/", "-")
    out: Path = HERE / "results" / f"{slug}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    total: int = len(results)
    exact: int = sum(1 for r in results if r["exact"])
    amounts: int = sum(1 for r in results if r["amounts"])
    answers: int = sum(1 for r in results if r["answered"])
    input_tokens: int = sum(int(str(r["input_tokens"])) for r in results)
    output_tokens: int = sum(int(str(r["output_tokens"])) for r in results)
    print(
        f"{settings.llm_model}: exact {exact}/{total}, amounts {amounts}/{total}, answers {answers}/{total}"
    )
    print(f"tokens: {input_tokens} in, {output_tokens} out → {out}")


if __name__ == "__main__":
    main()
