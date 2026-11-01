"""Telegram over long polling: no public IP, domain or VPS — the bot runs on a laptop at home."""

import logging

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message

from expense_tracker.config import Settings

log: logging.Logger = logging.getLogger(__name__)


def build_dispatcher(settings: Settings) -> Dispatcher:
    dp: Dispatcher = Dispatcher()
    # Messages from anyone outside the whitelist are dropped without a reply.
    dp.message.filter(F.from_user.id.in_(settings.allowed_user_ids))

    @dp.message()
    async def placeholder(message: Message) -> None:
        await message.answer("Бот запущен, агент трат ещё не подключён.")

    return dp


async def run(settings: Settings) -> None:
    log.info("model %s", settings.llm_model)
    bot: Bot = Bot(settings.telegram_bot_token.get_secret_value())
    await build_dispatcher(settings).start_polling(bot)
