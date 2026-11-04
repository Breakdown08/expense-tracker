"""Telegram over long polling: no public IP, domain or VPS — the bot runs on a laptop at home."""

import asyncio
import logging
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import date, datetime
from typing import Any, cast, override
from zoneinfo import ZoneInfo

import httpx
from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    Chat,
    ForceReply,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    TelegramObject,
)
from aiogram.utils.chat_action import ChatActionSender
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command as Resume
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from expense_tracker.agent import Agent, Context, build_agent, reply
from expense_tracker.config import Settings
from expense_tracker.db import make_engine, make_sessions
from expense_tracker.platforms import publishers
from expense_tracker.post import Action, Answer, PostContext, PostGraph, Publisher, build_post_graph, pause
from expense_tracker.rates import BankOfRussia, RateSource

log: logging.Logger = logging.getLogger(__name__)

TELEGRAM_LIMIT: int = 4096  # characters in one message; Telegram rejects longer ones

GREETING: str = """Пиши траты как в тетрадку: «фо 60к», «grab 45k, кофе вчера 35к».
Поправить: «нет, было 50». Спросить: «сколько за неделю?», «что записано сегодня».
/post — пост о неделе: сначала покажу черновик, без твоей кнопки ничего не выйдет."""

FAILED: str = "Модель не ответила. Попробуй позже; что успело записаться — спроси «что записано сегодня»."


NOTHING_TO_POST: str = "За неделю нет трат в VND: писать пост не о чем."
ASK_COMMENT: str = "Что поправить в посте?"
POST_MARK: re.Pattern[str] = re.compile(r"\[(post:\S+)\]")


def post_config(post: str) -> RunnableConfig:
    return {"configurable": {"thread_id": post}}


def buttons(post: str, *choices: tuple[str, Action]) -> InlineKeyboardMarkup:
    """The flow's id travels in the button: a post waiting since yesterday still finds its thread."""
    row: list[InlineKeyboardButton] = [
        InlineKeyboardButton(text=label, callback_data=f"{action}|{post}") for label, action in choices
    ]
    return InlineKeyboardMarkup(inline_keyboard=[row])


def finished(state: dict[str, Any], platforms: list[Publisher]) -> str:
    """The closing line once the flow has ended; the draft above it stays in the chat."""
    if state.get("action") == "cancel":
        return "Отменил."
    if not state.get("figures"):
        return NOTHING_TO_POST
    if not platforms:
        return "Площадки не настроены: скопируй текст выше и опубликуй сам."
    done: dict[str, str] = state.get("published", {})
    lines: list[str] = [f"{name}: {where}" for name, where in done.items()] or ["Ничего не вышло."]
    failed: dict[str, str] = state.get("failed", {})
    return "\n".join([*lines, *(f"{name}: не вышло, {why}" for name, why in failed.items())])


def signed(message: Message) -> str:
    """The group is one memory thread for two: the model needs to know who said what."""
    assert message.from_user is not None and message.text is not None
    return f"[{message.from_user.first_name}] {message.text}"


def chunks(text: str, size: int = TELEGRAM_LIMIT) -> list[str]:
    """Split at line breaks so that a month's list arrives whole; a single overlong line is cut."""
    parts: list[str] = []
    current: str = ""
    line: str
    for line in text.splitlines():
        while len(line) > size:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:size])
            line = line[size:]
        if current and len(current) + 1 + len(line) > size:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts or ["…"]


class InTurn(BaseMiddleware):
    """A chat's updates are handled one at a time, in the order they came.

    Two quick messages would otherwise run the agent on one memory thread at once, and the
    second turn vanished from the history. The lock is taken before the filters: aiogram runs
    sync filters in threads, so by the handler two messages may already be out of order.
    """

    def __init__(self) -> None:
        self.locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    @override
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        chat: Chat | None = data.get("event_chat")
        if chat is None:
            return await handler(event, data)
        async with self.locks[chat.id]:
            return await handler(event, data)


def build_dispatcher(
    settings: Settings,
    agent: Agent,
    sessions: async_sessionmaker[AsyncSession],
    rates: RateSource,
    post_graph: PostGraph,
    model: BaseChatModel,
    platforms: list[Publisher],
) -> Dispatcher:
    dp: Dispatcher = Dispatcher()
    dp.update.outer_middleware(InTurn())  # after aiogram's own, which sets event_chat
    # Messages from anyone outside the whitelist are dropped without a reply.
    dp.message.filter(F.from_user.id.in_(settings.allowed_user_ids))
    dp.callback_query.filter(F.from_user.id.in_(settings.allowed_user_ids))
    zone: ZoneInfo = ZoneInfo(settings.timezone)

    def post_context() -> PostContext:
        return PostContext(
            sessions=sessions, model=model, publishers=platforms, today=datetime.now(zone).date()
        )

    async def advance(message: Message, post: str, step: object) -> None:
        """Run the flow up to its next pause and show that pause, or how it ended."""
        config: RunnableConfig = post_config(post)
        try:
            async with ChatActionSender.typing(message.chat.id, cast(Bot, message.bot)):
                await post_graph.ainvoke(cast(Any, step), config, context=post_context())
        except Exception:
            log.exception("post flow failed")
            await message.answer("Сценарий поста сорвался. Запусти /post заново.")
            return
        waiting: dict[str, Any] | None = await pause(post_graph, config)
        if waiting is None:
            state: dict[str, Any] = (await post_graph.aget_state(config)).values
            await message.answer(finished(state, platforms))
        elif waiting["step"] == "review":
            draft: str = waiting["draft"]
            await message.answer(
                f"{draft}\n\n({len(draft)} знаков)",
                reply_markup=buttons(
                    post, ("Опубликовать", "publish"), ("Править", "edit"), ("Отмена", "cancel")
                ),
            )
        else:
            failed: str = "\n".join(f"{name}: {why}" for name, why in waiting["failed"].items())
            await message.answer(
                f"Не вышло:\n{failed}",
                reply_markup=buttons(post, ("Повторить", "retry"), ("Хватит", "stop")),
            )

    @dp.message(Command("post"))
    async def on_post(message: Message) -> None:
        post: str = f"post:{message.chat.id}:{message.message_id}"
        await advance(message, post, {})

    @dp.callback_query(F.data.regexp(r"^(publish|edit|cancel|retry|stop)\|post:"))
    async def on_post_button(query: CallbackQuery) -> None:
        assert query.data is not None and isinstance(query.message, Message)
        action: str
        post: str
        action, post = query.data.split("|", 1)
        await query.answer()
        await query.message.edit_reply_markup(reply_markup=None)  # one press, one decision
        if await pause(post_graph, post_config(post)) is None:
            await query.message.answer("Этот пост уже закрыт.")
        elif action == "edit":
            await query.message.answer(f"{ASK_COMMENT} [{post}]", reply_markup=ForceReply())
        else:
            await advance(query.message, post, Resume(resume=Answer(action=cast(Action, action))))

    @dp.message(F.text, F.reply_to_message.text.contains(ASK_COMMENT))
    async def on_post_comment(message: Message) -> None:
        assert message.reply_to_message is not None and message.reply_to_message.text is not None
        mark: re.Match[str] | None = POST_MARK.search(message.reply_to_message.text)
        if mark is None or await pause(post_graph, post_config(mark[1])) is None:
            await message.answer("Этот пост уже закрыт.")
            return
        answer: Answer = Answer(action="edit", comment=message.text or "")
        await advance(message, mark[1], Resume(resume=answer))

    @dp.message(CommandStart())
    async def on_start(message: Message) -> None:
        await message.answer(GREETING)

    @dp.message(F.text)
    async def on_text(message: Message, bot: Bot) -> None:
        assert message.from_user is not None and message.text is not None  # the filters above
        answer: str
        async with ChatActionSender.typing(message.chat.id, bot):
            today: date = datetime.now(zone).date()
            context: Context = Context(
                user_id=message.from_user.id, sessions=sessions, rates=rates, today=today
            )
            try:
                answer = await reply(agent, message.chat.id, signed(message), context)
            except Exception:
                log.exception("agent failed")
                answer = FAILED
        part: str
        for part in chunks(answer):
            await message.answer(part)

    @dp.message()
    async def on_other(message: Message) -> None:
        await message.answer("Пока понимаю только текст.")

    return dp


async def run(settings: Settings) -> None:
    # No temperature: recent Claude models reject the parameter, and gpt-oss is meant to run at its default.
    model: BaseChatModel = init_chat_model(
        settings.llm_model, api_key=settings.llm_api_key.get_secret_value()
    )
    engine: AsyncEngine = make_engine(settings.database_url)
    saver: AsyncPostgresSaver
    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
        await saver.setup()  # LangGraph's own tables; idempotent
        agent: Agent = build_agent(model, saver)
        log.info("model %s", settings.llm_model)
        bot: Bot = Bot(settings.telegram_bot_token.get_secret_value())
        client: httpx.AsyncClient
        async with httpx.AsyncClient(timeout=30) as client:
            platforms: list[Publisher] = publishers(settings, bot, client)
            log.info("post platforms: %s", [p.name for p in platforms])
            dispatcher: Dispatcher = build_dispatcher(
                settings,
                agent,
                make_sessions(engine),
                BankOfRussia(),
                build_post_graph(saver),
                model,
                platforms,
            )
            try:
                await dispatcher.start_polling(bot)
            finally:
                await engine.dispose()
