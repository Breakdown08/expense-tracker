"""The posting flow: pauses, the edit loop, and never publishing the same post twice."""

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from expense_tracker.expenses import Category, CategoryTotal, NewExpense, Period, add_expenses
from expense_tracker.platforms import THREADS_LIMIT, Threads
from expense_tracker.post import PostContext, PostGraph, Publisher, build_post_graph, describe, pause
from tests.conftest import requires_db
from tests.scripted import scripted

TODAY: date = date(2026, 11, 8)
CONFIG: RunnableConfig = {"configurable": {"thread_id": "post-1"}}


class Recording:
    """A platform that remembers what it got; fails while `broken` is set."""

    def __init__(self, name: str, broken: bool = False) -> None:
        self.name: str = name
        self.broken: bool = broken
        self.posts: list[str] = []

    async def publish(self, text: str) -> str:
        if self.broken:
            raise ConnectionError("down")
        self.posts.append(text)
        return f"{self.name} ok"


def total(category: Category, now: int, before: int, currency: str = "VND") -> CategoryTotal:
    return CategoryTotal(
        category=category, currency=currency, total=Decimal(now), count=1, previous_total=Decimal(before)
    )


def test_figures_have_shares_and_no_amounts() -> None:
    week: Period = Period(start=date(2026, 11, 2), end=date(2026, 11, 9))
    text: str = describe(
        [
            total(Category.GROCERIES, 750000, 500000),
            total(Category.HEALTH, 250000, 0),
            total(Category.TRAVEL, 0, 500000),
            total(Category.TRAVEL, 40, 0, "USD"),
        ],
        week,
    )
    assert text.splitlines() == [
        "Week 2026-11-02 .. 2026-11-08",
        "Total compared to the week before: +0%",
        "groceries: 75% of spending, +50%",
        "health: 25% of spending, not there before",
    ]
    assert describe([], week) == ""


async def seed(sessions: async_sessionmaker[AsyncSession]) -> None:
    session: AsyncSession
    async with sessions() as session:
        await add_expenses(
            session,
            1,
            [NewExpense(amount=Decimal(60000), category=Category.GROCERIES, spent_on=date(2026, 11, 5))],
        )


def context(
    sessions: async_sessionmaker[AsyncSession], publishers: Sequence[Publisher], *drafts: str
) -> PostContext:
    return PostContext(
        sessions=sessions,
        model=scripted(*[AIMessage(d) for d in drafts]),
        publishers=publishers,
        today=TODAY,
    )


@requires_db
async def test_nothing_spent_nothing_to_approve(sessions: async_sessionmaker[AsyncSession]) -> None:
    graph: PostGraph = build_post_graph(InMemorySaver())
    await graph.ainvoke({}, CONFIG, context=context(sessions, []))
    assert await pause(graph, CONFIG) is None


@requires_db
async def test_edit_loop_then_publish(sessions: async_sessionmaker[AsyncSession]) -> None:
    await seed(sessions)
    channel: Recording = Recording("Telegram")
    ctx: PostContext = context(sessions, [channel], "первый", "второй")
    graph: PostGraph = build_post_graph(InMemorySaver())
    await graph.ainvoke({}, CONFIG, context=ctx)
    assert await pause(graph, CONFIG) == {"step": "review", "draft": "первый"}
    await graph.ainvoke(Command(resume={"action": "edit", "comment": "короче"}), CONFIG, context=ctx)
    assert await pause(graph, CONFIG) == {"step": "review", "draft": "второй"}
    assert channel.posts == []  # nothing goes out before a person says so
    await graph.ainvoke(Command(resume={"action": "publish"}), CONFIG, context=ctx)
    assert channel.posts == ["второй"]
    assert await pause(graph, CONFIG) is None


@requires_db
async def test_cancel_publishes_nothing(sessions: async_sessionmaker[AsyncSession]) -> None:
    await seed(sessions)
    channel: Recording = Recording("Telegram")
    ctx: PostContext = context(sessions, [channel], "пост")
    graph: PostGraph = build_post_graph(InMemorySaver())
    await graph.ainvoke({}, CONFIG, context=ctx)
    await graph.ainvoke(Command(resume={"action": "cancel"}), CONFIG, context=ctx)
    assert channel.posts == []
    assert await pause(graph, CONFIG) is None


@requires_db
async def test_retry_goes_only_to_the_platform_that_failed(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    await seed(sessions)
    threads: Recording = Recording("Threads", broken=True)
    channel: Recording = Recording("Telegram")
    ctx: PostContext = context(sessions, [threads, channel], "пост")
    graph: PostGraph = build_post_graph(InMemorySaver())
    await graph.ainvoke({}, CONFIG, context=ctx)
    await graph.ainvoke(Command(resume={"action": "publish"}), CONFIG, context=ctx)
    assert await pause(graph, CONFIG) == {
        "step": "failed",
        "published": {"Telegram": "Telegram ok"},
        "failed": {"Threads": "down"},
    }
    threads.broken = False
    await graph.ainvoke(Command(resume={"action": "retry"}), CONFIG, context=ctx)
    assert await pause(graph, CONFIG) is None
    assert (threads.posts, channel.posts) == (["пост"], ["пост"])  # the channel got it once


@requires_db
async def test_stop_after_a_failure_ends_the_flow(sessions: async_sessionmaker[AsyncSession]) -> None:
    await seed(sessions)
    ctx: PostContext = context(sessions, [Recording("Threads", broken=True)], "пост")
    graph: PostGraph = build_post_graph(InMemorySaver())
    await graph.ainvoke({}, CONFIG, context=ctx)
    await graph.ainvoke(Command(resume={"action": "publish"}), CONFIG, context=ctx)
    await graph.ainvoke(Command(resume={"action": "stop"}), CONFIG, context=ctx)
    assert await pause(graph, CONFIG) is None


@requires_db
async def test_a_paused_post_survives_a_restart(
    sessions: async_sessionmaker[AsyncSession], test_db_url: str
) -> None:
    """Two connections to Postgres stand for the process before and after a reboot."""
    await seed(sessions)
    channel: Recording = Recording("Telegram")
    ctx: PostContext = context(sessions, [channel], "пост")
    config: RunnableConfig = {"configurable": {"thread_id": f"post-{uuid4()}"}}
    saver: AsyncPostgresSaver
    async with AsyncPostgresSaver.from_conn_string(test_db_url) as saver:
        await saver.setup()
        await build_post_graph(saver).ainvoke({}, config, context=ctx)
    async with AsyncPostgresSaver.from_conn_string(test_db_url) as saver:
        graph: PostGraph = build_post_graph(saver)
        assert await pause(graph, config) == {"step": "review", "draft": "пост"}
        await graph.ainvoke(Command(resume={"action": "publish"}), config, context=ctx)
    assert channel.posts == ["пост"]


def threads_with(handler: httpx.MockTransport) -> Threads:
    return Threads("secret", "42", httpx.AsyncClient(transport=handler))


async def test_threads_makes_a_container_then_publishes_it() -> None:
    paths: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"id": "7" if request.url.path.endswith("threads") else "9"})

    assert await threads_with(httpx.MockTransport(answer)).publish("пост") == "post 9"
    assert paths == ["/v1.0/42/threads", "/v1.0/42/threads_publish"]


async def test_threads_error_keeps_the_token_out() -> None:
    bad: httpx.MockTransport = httpx.MockTransport(lambda r: httpx.Response(400, text="expired"))
    error: RuntimeError | None = None
    try:
        await threads_with(bad).publish("пост")
    except RuntimeError as caught:
        error = caught
    assert error is not None
    assert "expired" in str(error) and "secret" not in str(error)


async def test_threads_refuses_a_long_post_without_calling() -> None:
    def never(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call")

    with pytest.raises(ValueError, match="characters"):
        await threads_with(httpx.MockTransport(never)).publish("я" * (THREADS_LIMIT + 1))
