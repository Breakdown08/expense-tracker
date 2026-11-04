"""The bot against a Telegram that records requests instead of sending them; the agent runs on a script."""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, override
from zoneinfo import ZoneInfo

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.methods import SendMessage, TelegramMethod
from aiogram.types import (
    CallbackQuery,
    Chat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    PhotoSize,
    Update,
    User,
)
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from expense_tracker.agent import Agent, build_agent, thread
from expense_tracker.bot import (
    ASK_COMMENT,
    FAILED,
    GREETING,
    NOTHING_TO_POST,
    TELEGRAM_LIMIT,
    build_dispatcher,
    chunks,
)
from expense_tracker.config import Settings
from expense_tracker.expenses import Category, NewExpense, add_expenses
from expense_tracker.post import Publisher, build_post_graph
from tests.conftest import FixedRates, requires_db
from tests.scripted import Script, ScriptedModel, scripted
from tests.test_post import Recording

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


class Watching(ScriptedModel):
    """Keeps the messages of every call."""

    seen: list[list[BaseMessage]] = []

    @override
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(messages)
        return await super()._agenerate(messages, stop, run_manager, **kwargs)


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token=SecretStr("42:TEST"),
        allowed_user_ids=[ME, WIFE],
        groq_api_key=SecretStr("test"),
    )


def local_day() -> date:
    return datetime.now(ZoneInfo(settings().timezone)).date()


async def spend(sessions: async_sessionmaker[AsyncSession]) -> None:
    """Something for this month's report: the flow's date is the real one here."""
    session: AsyncSession
    async with sessions() as session:
        item: NewExpense = NewExpense(
            amount=Decimal(60000), category=Category.GROCERIES, spent_on=local_day()
        )
        await add_expenses(session, ME, [item])


def today(chat_id: int) -> str:
    """The thread the bot writes to now: the chat's day in the family's time zone."""
    return thread(chat_id, datetime.now(ZoneInfo(settings().timezone)).date())


class Telegram:
    """A dispatcher, a bot on the recorder, and updates numbered as Telegram does."""

    def __init__(
        self,
        agent: Agent,
        sessions: async_sessionmaker[AsyncSession] | None = None,
        model: BaseChatModel | None = None,
        platforms: list[Publisher] | None = None,
    ) -> None:
        self.session: Recorder = Recorder()
        self.bot: Bot = Bot("42:TEST", session=self.session)
        self.dp: Dispatcher = build_dispatcher(
            settings(),
            agent,
            sessions or async_sessionmaker(),  # no tool runs in most tests: never opened
            FixedRates(),
            build_post_graph(InMemorySaver()),
            model or scripted(),
            platforms or [],
        )
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

    async def reply(self, user: int, to: str, text: str) -> None:
        """Answer the bot's message, as ForceReply makes a person do."""
        self.update_id += 1
        chat: Chat = Chat(id=user, type="private")
        now: datetime = datetime.now(UTC)
        message: Message = Message(
            message_id=self.update_id,
            date=now,
            chat=chat,
            from_user=User(id=user, is_bot=False, first_name=NAMES[user]),
            text=text,
            reply_to_message=Message(message_id=1, date=now, chat=chat, text=to),
        )
        await self.dp.feed_update(self.bot, Update(update_id=self.update_id, message=message))

    async def press(self, user: int, data: str) -> None:
        self.update_id += 1
        chat: Chat = Chat(id=user, type="private")
        query: CallbackQuery = CallbackQuery(
            id=str(self.update_id),
            from_user=User(id=user, is_bot=False, first_name=NAMES[user]),
            chat_instance="c",
            data=data,
            message=Message(message_id=1, date=datetime.now(UTC), chat=chat, text="…"),
        )
        await self.dp.feed_update(self.bot, Update(update_id=self.update_id, callback_query=query))

    def button(self, label: str) -> str:
        """The data of the latest button with this label."""
        method: SendMessage
        for method in reversed([m for m in self.session.requests if isinstance(m, SendMessage)]):
            markup: object = method.reply_markup
            if isinstance(markup, InlineKeyboardMarkup):
                button: InlineKeyboardButton
                for button in markup.inline_keyboard[0]:
                    if button.text == label:
                        return str(button.callback_data)
        raise AssertionError(f"no button {label}")


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


@requires_db
async def test_post_from_command_to_channel(sessions: async_sessionmaker[AsyncSession]) -> None:
    await spend(sessions)
    channel: Recording = Recording("Telegram")
    model: ScriptedModel = scripted(AIMessage("Первый."), AIMessage("Второй."))
    telegram: Telegram = Telegram(build_agent(scripted()), sessions, model, [channel])
    await telegram.send(ME, "/отчёт")
    assert telegram.session.sent() == ["Первый.\n\n(7 знаков)"]
    await telegram.press(ME, telegram.button("Править"))
    prompt: str = telegram.session.sent()[-1]
    assert prompt.startswith(ASK_COMMENT)
    await telegram.reply(ME, prompt, "короче")
    assert telegram.session.sent()[-1] == "Второй.\n\n(7 знаков)"
    assert channel.posts == []
    published: str = telegram.button("Опубликовать")
    await telegram.press(ME, published)
    assert channel.posts == ["Второй."]
    assert telegram.session.sent()[-1] == "Telegram: Telegram ok"
    await telegram.press(WIFE, published)  # an old button does not post again
    assert channel.posts == ["Второй."]
    assert telegram.session.sent()[-1] == "Этот пост уже закрыт."


@requires_db
async def test_post_failure_offers_a_retry(sessions: async_sessionmaker[AsyncSession]) -> None:
    await spend(sessions)
    channel: Recording = Recording("Telegram", broken=True)
    telegram: Telegram = Telegram(build_agent(scripted()), sessions, scripted(AIMessage("Пост.")), [channel])
    await telegram.send(ME, "/отчёт")
    await telegram.press(ME, telegram.button("Опубликовать"))
    assert telegram.session.sent()[-1] == "Не вышло:\nTelegram: down"
    channel.broken = False
    await telegram.press(ME, telegram.button("Повторить"))
    assert channel.posts == ["Пост."]


@requires_db
async def test_post_without_platforms_is_left_to_copy(sessions: async_sessionmaker[AsyncSession]) -> None:
    await spend(sessions)
    telegram: Telegram = Telegram(build_agent(scripted()), sessions, scripted(AIMessage("Пост.")))
    await telegram.send(ME, "/отчёт")
    await telegram.press(ME, telegram.button("Опубликовать"))
    assert telegram.session.sent()[-1].startswith("Площадки не настроены")


@requires_db
async def test_post_cancelled_and_stranger_cannot_press(sessions: async_sessionmaker[AsyncSession]) -> None:
    await spend(sessions)
    channel: Recording = Recording("Telegram")
    telegram: Telegram = Telegram(build_agent(scripted()), sessions, scripted(AIMessage("Пост.")), [channel])
    await telegram.send(ME, "/отчёт")
    await telegram.press(STRANGER, telegram.button("Опубликовать"))
    assert channel.posts == []
    await telegram.press(ME, telegram.button("Отмена"))
    assert telegram.session.sent()[-1] == "Отменил."


@requires_db
async def test_post_with_nothing_spent(sessions: async_sessionmaker[AsyncSession]) -> None:
    telegram: Telegram = Telegram(build_agent(scripted()), sessions)
    await telegram.send(ME, "/отчёт")
    assert telegram.session.sent() == [NOTHING_TO_POST]


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


@requires_db
@pytest.mark.parametrize("command", ["/сегодня", "/СЕГОДНЯ", "/today", "/неделя", "/week"])
async def test_a_command_never_reaches_the_model(
    command: str, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await spend(sessions)
    script: Script = Script([AIMessage("never reached")])
    agent: Agent = build_agent(ScriptedModel(messages=script), InMemorySaver())
    telegram: Telegram = Telegram(agent, sessions)
    await telegram.send(ME, command)
    assert script.answers == [AIMessage("never reached")]
    assert "Итого: 60 000 ₫" in telegram.session.sent()[0]


@requires_db
async def test_the_list_of_a_command_is_there_for_the_next_message(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    await spend(sessions)
    model: Watching = Watching(messages=Script([AIMessage("Какую запись?")]))
    telegram: Telegram = Telegram(build_agent(model, InMemorySaver()), sessions)
    await telegram.send(ME, "/сегодня")
    assert "#" not in telegram.session.sent()[0] and "1. 60 000 ₫" in telegram.session.sent()[0]
    await telegram.send(ME, "удали первую")
    seen: list[str] = [str(m.content) for m in model.seen[0]]
    assert seen[-3] == "[Кирилл] /сегодня"
    assert "1. 60 000 ₫" in seen[-2] and "Record ids, in the order above: #" in seen[-2]
    assert seen[-1] == "[Кирилл] удали первую"


@requires_db
async def test_yesterday_is_empty_when_all_was_spent_today(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    await spend(sessions)
    telegram: Telegram = Telegram(build_agent(scripted()), sessions)
    await telegram.send(ME, "/вчера")
    assert telegram.session.sent()[0].endswith("трат нет")


async def test_a_command_from_a_stranger_is_dropped() -> None:
    telegram: Telegram = Telegram(build_agent(scripted()))
    await telegram.send(STRANGER, "/сегодня")
    assert telegram.session.sent() == []


async def test_the_next_turn_goes_to_the_model_without_past_thinking() -> None:
    idea: dict[str, str] = {"type": "thinking", "thinking": "…", "signature": "made under another prompt"}
    said: dict[str, str] = {"type": "text", "text": "Привет"}
    model: Watching = Watching(messages=Script([AIMessage(content=[idea, said]), AIMessage("Ещё раз")]))
    telegram: Telegram = Telegram(build_agent(model, InMemorySaver()))
    await telegram.send(ME, "привет")
    await telegram.send(ME, "ещё раз")
    assert model.seen[1][-2].content == [said]
