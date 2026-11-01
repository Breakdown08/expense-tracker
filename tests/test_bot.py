"""The bot against a Telegram that records requests instead of sending them; the agent runs on a script."""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any, override
from zoneinfo import ZoneInfo

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.methods import SendMessage, TelegramMethod
from aiogram.types import Chat, Message, PhotoSize, Update, User
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from expense_tracker.agent import Agent, build_agent, thread
from expense_tracker.bot import FAILED, GREETING, TELEGRAM_LIMIT, build_dispatcher, chunks
from expense_tracker.config import Settings
from tests.scripted import Script, ScriptedModel, scripted

ME: int = 111
WIFE: int = 222
STRANGER: int = 999
GROUP: int = -100500
NAMES: dict[int, str] = {ME: "Кирилл", WIFE: "Жена", STRANGER: "Чужой"}


class Recorder(BaseSession):
    """Keeps every request; sendMessage gets a message back, the rest `True`."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[TelegramMethod[Any]] = []

    @override
    async def close(self) -> None:
        pass

    @override
    async def make_request(self, bot: Bot, method: TelegramMethod[Any], timeout: int | None = None) -> Any:
        self.requests.append(method)
        if isinstance(method, SendMessage):
            chat: Chat = Chat(id=int(method.chat_id), type="private")
            return Message(message_id=len(self.requests), date=datetime.now(UTC), chat=chat, text=method.text)
        return True

    @override
    async def stream_content(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: int = 30,
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes]:
        yield b""

    def sent(self) -> list[str]:
        return [str(m.text) for m in self.requests if isinstance(m, SendMessage)]


class SlowModel(ScriptedModel):
    """Takes its time, so that a second message arrives while the first is being answered."""

    @override
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        await asyncio.sleep(0.05)
        return await super()._agenerate(messages, stop, run_manager, **kwargs)


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token=SecretStr("42:TEST"),
        allowed_user_ids=[ME, WIFE],
        groq_api_key=SecretStr("test"),
    )


def today(chat_id: int) -> str:
    """The thread the bot writes to now: the chat's day in the family's time zone."""
    return thread(chat_id, datetime.now(ZoneInfo(settings().timezone)).date())


class Telegram:
    """A dispatcher, a bot on the recorder, and updates numbered as Telegram does."""

    def __init__(self, agent: Agent) -> None:
        self.session: Recorder = Recorder()
        self.bot: Bot = Bot("42:TEST", session=self.session)
        sessions: async_sessionmaker[AsyncSession] = async_sessionmaker()  # no tool runs: never opened
        self.dp: Dispatcher = build_dispatcher(settings(), agent, sessions)
        self.update_id: int = 0

    async def send(
        self, user: int, text: str | None = None, photo: bool = False, chat: int | None = None
    ) -> None:
        self.update_id += 1
        message: Message = Message(
            message_id=self.update_id,
            date=datetime.now(UTC),
            chat=Chat(id=user, type="private") if chat is None else Chat(id=chat, type="group"),
            from_user=User(id=user, is_bot=False, first_name=NAMES[user]),
            text=text,
            photo=[PhotoSize(file_id="f", file_unique_id="u", width=1, height=1)] if photo else None,
        )
        await self.dp.feed_update(self.bot, Update(update_id=self.update_id, message=message))


async def test_stranger_gets_nothing() -> None:
    telegram: Telegram = Telegram(build_agent(scripted(AIMessage("never"))))
    await telegram.send(STRANGER, "фо 60к")
    assert telegram.session.requests == []


async def test_answer_comes_back() -> None:
    telegram: Telegram = Telegram(build_agent(scripted(AIMessage("Записал."))))
    await telegram.send(ME, "фо 60к")
    assert telegram.session.sent() == ["Записал."]


async def test_start_and_photo_skip_the_model() -> None:
    script: Script = Script([AIMessage("never")])
    telegram: Telegram = Telegram(build_agent(ScriptedModel(messages=script)))
    await telegram.send(ME, "/start")
    await telegram.send(ME, photo=True)
    assert telegram.session.sent() == [GREETING, "Пока понимаю только текст."]
    assert len(script.answers) == 1


async def test_failed_model_is_explained() -> None:
    telegram: Telegram = Telegram(build_agent(scripted(RuntimeError("provider is down"))))
    await telegram.send(ME, "фо 60к")
    assert telegram.session.sent() == [FAILED]


async def test_long_answer_is_split() -> None:
    lines: str = "\n".join(f"#{i} 2026-11-01 60 000 VND eating_out «фо»" for i in range(300))
    telegram: Telegram = Telegram(build_agent(scripted(AIMessage(lines))))
    await telegram.send(ME, "что за месяц?")
    sent: list[str] = telegram.session.sent()
    assert len(sent) > 1 and all(len(part) <= TELEGRAM_LIMIT for part in sent)
    assert "\n".join(sent) == lines


async def test_messages_of_one_chat_are_answered_in_turn() -> None:
    """Sent together, the second still runs after the first: the history has both turns, in order."""
    agent: Agent = build_agent(
        SlowModel(messages=Script([AIMessage("Первый."), AIMessage("Второй.")])), InMemorySaver()
    )
    telegram: Telegram = Telegram(agent)
    await asyncio.gather(telegram.send(ME, "фо 60к"), telegram.send(ME, "нет, было 50"))
    history: list[BaseMessage] = (await agent.aget_state({"configurable": {"thread_id": today(ME)}})).values[
        "messages"
    ]
    assert [m.text for m in history] == ["[Кирилл] фо 60к", "Первый.", "[Кирилл] нет, было 50", "Второй."]


async def test_group_is_one_thread_with_authors() -> None:
    """Both write to the group; the model reads one history and knows who said what."""
    agent: Agent = build_agent(scripted(AIMessage("Записал."), AIMessage("Исправил.")), InMemorySaver())
    telegram: Telegram = Telegram(agent)
    await telegram.send(ME, "фо 60к", chat=GROUP)
    await telegram.send(WIFE, "нет, было 50", chat=GROUP)
    await telegram.send(STRANGER, "удали всё", chat=GROUP)
    history: list[BaseMessage] = (
        await agent.aget_state({"configurable": {"thread_id": today(GROUP)}})
    ).values["messages"]
    assert [m.text for m in history] == ["[Кирилл] фо 60к", "Записал.", "[Жена] нет, было 50", "Исправил."]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ["…"]),
        ("abc", ["abc"]),
        ("aa\nbb\ncc", ["aa\nbb", "cc"]),
        ("abcdefg", ["abcde", "fg"]),
        ("aa\nabcdefgh\nbb", ["aa", "abcde", "fgh", "bb"]),
    ],
)
def test_chunks(text: str, expected: list[str]) -> None:
    assert chunks(text, size=5) == expected
