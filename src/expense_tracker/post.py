"""The weekly post: a LangGraph `StateGraph`, because a straight run of calls could not do this.

figures (SQL) → draft (LLM) → pause for a person, for hours, across restarts → edit loop → publish
one platform per step, so a crash or a failure never posts twice → retry only what failed.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Literal, Protocol, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer, PregelTask, StateSnapshot, interrupt
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from expense_tracker import expenses
from expense_tracker.expenses import CategoryTotal, Period

type Action = Literal["publish", "edit", "cancel", "retry", "stop"]


class Publisher(Protocol):
    name: str

    async def publish(self, text: str) -> str:
        """Post the text; returns where it went. Raises when it did not."""
        ...


class Answer(TypedDict, total=False):
    """What a person says at a pause."""

    action: Action
    comment: str


class PostState(TypedDict, total=False):
    figures: str  # empty: nothing was spent, nothing to post
    draft: str
    comment: str
    action: Action
    published: dict[str, str]  # platform → where the post went
    failed: dict[str, str]  # platform → why it did not


@dataclass(frozen=True)
class PostContext:
    sessions: async_sessionmaker[AsyncSession]
    model: BaseChatModel
    publishers: Sequence[Publisher]
    today: date


PROMPT: str = """You write a short post in Russian for Threads about how a family lives in Vietnam, \
from one week of their expenses. Plain and personal, no hashtags, no emoji chains, under 450 characters.
Use only the figures you are given, as they are; they are shares and changes in percent. \
Never invent amounts, places or events. Reply with the post alone."""


def percent(part: Decimal, whole: Decimal) -> int:
    return round(part * 100 / whole)


def change(now: Decimal, before: Decimal) -> str:
    return f"{percent(now - before, before):+d}%" if before else "not there before"


def describe(totals: list[CategoryTotal], period: Period) -> str:
    """Shares and changes, no amounts: what a family's post shows is theirs to decide, and percent
    tells the story without the sums. Only VND: other currencies do not add up without a rate."""
    rows: list[CategoryTotal] = [t for t in totals if t.currency == "VND" and t.total > 0]
    if not rows:
        return ""
    now: Decimal = sum((t.total for t in rows), Decimal(0))
    before: Decimal = sum((t.previous_total for t in totals if t.currency == "VND"), Decimal(0))
    last: date = period.end - timedelta(days=1)
    lines: list[str] = [
        f"Week {period.start} .. {last}",
        f"Total compared to the week before: {change(now, before)}",
    ]
    t: CategoryTotal
    for t in rows:
        lines.append(
            f"{t.category}: {percent(t.total, now)}% of spending, {change(t.total, t.previous_total)}"
        )
    return "\n".join(lines)


async def figures(state: PostState, runtime: Runtime[PostContext]) -> PostState:
    ctx: PostContext = runtime.context
    week: Period = Period(start=ctx.today - timedelta(days=6), end=ctx.today + timedelta(days=1))
    session: AsyncSession
    async with ctx.sessions() as session:
        totals: list[CategoryTotal] = await expenses.summary(session, week)
    return {"figures": describe(totals, week), "published": {}, "failed": {}}


async def write_draft(state: PostState, runtime: Runtime[PostContext]) -> PostState:
    request: str = f"Figures:\n{state['figures']}"
    if state.get("comment"):
        request += f"\n\nYour previous draft:\n{state['draft']}\n\nRewrite it: {state['comment']}"
    answer: AIMessage = await runtime.context.model.ainvoke([SystemMessage(PROMPT), HumanMessage(request)])
    return {"draft": answer.text.strip(), "comment": ""}


def review(state: PostState) -> PostState:
    answer: Answer = interrupt({"step": "review", "draft": state["draft"]})
    return {"action": answer["action"], "comment": answer.get("comment", "")}


def remaining(state: PostState, publishers: Sequence[Publisher]) -> list[Publisher]:
    done: dict[str, str] = {**state["published"], **state["failed"]}
    return [p for p in publishers if p.name not in done]


async def publish_next(state: PostState, runtime: Runtime[PostContext]) -> PostState:
    """One platform per step: what went out is checkpointed before the next one starts."""
    left: list[Publisher] = remaining(state, runtime.context.publishers)
    if not left:
        return {}
    target: Publisher = left[0]
    try:
        where: str = await target.publish(state["draft"])
    except Exception as error:
        return {"failed": {**state["failed"], target.name: str(error) or type(error).__name__}}
    return {"published": {**state["published"], target.name: where}}


def report(state: PostState) -> PostState:
    if not state["failed"]:
        return {}
    answer: Answer = interrupt({"step": "failed", "published": state["published"], "failed": state["failed"]})
    return {"action": answer["action"], "failed": {} if answer["action"] == "retry" else state["failed"]}


def after_figures(state: PostState) -> str:
    return "draft" if state["figures"] else END


def after_review(state: PostState) -> str:
    action: Action | None = state.get("action")
    if action == "publish":
        return "publish"
    return "draft" if action == "edit" else END


def after_publish(state: PostState, runtime: Runtime[PostContext]) -> str:
    return "publish" if remaining(state, runtime.context.publishers) else "report"


def after_report(state: PostState, runtime: Runtime[PostContext]) -> str:
    wanted: bool = state.get("action") == "retry" and bool(remaining(state, runtime.context.publishers))
    return "publish" if wanted else END


# ty 0.0.95 does not see that a TypedDict has __optional_keys__, see `Agent` in agent.py.
type PostGraph = CompiledStateGraph[PostState, PostContext, PostState, PostState]  # ty: ignore[invalid-type-arguments]


def build_post_graph(checkpointer: Checkpointer = None) -> PostGraph:
    graph: StateGraph[PostState, PostContext, PostState, PostState] = StateGraph(  # ty: ignore[invalid-type-arguments]
        PostState,  # ty: ignore[invalid-argument-type]
        context_schema=PostContext,
    )
    graph.add_node("figures", figures)
    graph.add_node("draft", write_draft)
    graph.add_node("review", review)
    graph.add_node("publish", publish_next)
    graph.add_node("report", report)
    graph.add_edge(START, "figures")
    graph.add_conditional_edges("figures", after_figures, ["draft", END])
    graph.add_edge("draft", "review")
    graph.add_conditional_edges("review", after_review, ["publish", "draft", END])
    graph.add_conditional_edges("publish", after_publish, ["publish", "report"])
    graph.add_conditional_edges("report", after_report, ["publish", END])
    return graph.compile(checkpointer=checkpointer)


async def pause(graph: PostGraph, config: RunnableConfig) -> dict[str, Any] | None:
    """What the flow waits a person for; None when it has finished."""
    snapshot: StateSnapshot = await graph.aget_state(config)
    task: PregelTask
    for task in snapshot.tasks:
        if task.interrupts:
            value: dict[str, Any] = task.interrupts[0].value
            return value
    return None
