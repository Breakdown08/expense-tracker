"""Telegram over long polling: no public IP, domain or VPS — the bot runs on a laptop at home."""

import logging

import psycopg
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message

from expense_tracker.config import Settings

log: logging.Logger = logging.getLogger(__name__)


async def check_database(url: str) -> None:
    """Fail at start, not on the first expense, when Postgres is unreachable."""
    conn: psycopg.AsyncConnection[tuple[object, ...]]
    async with await psycopg.AsyncConnection.connect(url) as conn:
        await conn.execute("SELECT 1")


def build_dispatcher(settings: Settings) -> Dispatcher:
    dp: Dispatcher = Dispatcher()
    # Messages from anyone outside the whitelist are dropped without a reply.
    dp.message.filter(F.from_user.id.in_(settings.allowed_user_ids))

    @dp.message()
    async def placeholder(message: Message) -> None:
        await message.answer("Бот запущен, агент трат ещё не подключён.")

    return dp


async def run(settings: Settings) -> None:
    await check_database(settings.database_url)
    log.info("database reachable; model %s", settings.llm_model)
    bot: Bot = Bot(settings.telegram_bot_token.get_secret_value())
    await build_dispatcher(settings).start_polling(bot)
