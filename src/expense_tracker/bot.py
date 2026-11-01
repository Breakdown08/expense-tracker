"""Telegram over long polling: no public IP, domain or VPS — the bot runs on a laptop at home."""

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.types import Message
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from expense_tracker.agent import Agent, Context, build_agent, reply
from expense_tracker.config import Settings
from expense_tracker.db import make_engine, make_sessions

log: logging.Logger = logging.getLogger(__name__)


def build_dispatcher(
    settings: Settings, agent: Agent, sessions: async_sessionmaker[AsyncSession]
) -> Dispatcher:
    dp: Dispatcher = Dispatcher()
    # Messages from anyone outside the whitelist are dropped without a reply.
    dp.message.filter(F.from_user.id.in_(settings.allowed_user_ids))
    zone: ZoneInfo = ZoneInfo(settings.timezone)

    @dp.message(F.text)
    async def on_text(message: Message) -> None:
        assert message.from_user is not None and message.text is not None  # the filters above
        await message.chat.do(ChatAction.TYPING)
        today: date = datetime.now(zone).date()
        context: Context = Context(user_id=message.from_user.id, sessions=sessions, today=today)
        answer: str
        try:
            answer = await reply(agent, message.chat.id, message.text, context)
        except Exception:
            log.exception("agent failed")
            answer = (
                "Модель не ответила. Попробуй позже; что успело записаться — спроси «что записано сегодня»."
            )
        await message.answer(answer or "…")

    return dp


async def run(settings: Settings) -> None:
    model: BaseChatModel = init_chat_model(
        settings.llm_model, api_key=settings.llm_api_key.get_secret_value(), temperature=0
    )
    engine: AsyncEngine = make_engine(settings.database_url)
    saver: AsyncPostgresSaver
    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
        await saver.setup()  # LangGraph's own tables; idempotent
        agent: Agent = build_agent(model, saver)
        log.info("model %s", settings.llm_model)
        bot: Bot = Bot(settings.telegram_bot_token.get_secret_value())
        try:
            await build_dispatcher(settings, agent, make_sessions(engine)).start_polling(bot)
        finally:
            await engine.dispose()
