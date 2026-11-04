"""Templates of the commands and direct answers: no model anywhere."""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from expense_tracker.expenses import Category, NewExpense, add_expenses
from expense_tracker.models import Expense
from expense_tracker.rates import Rates
from expense_tracker.reports import Report, report, title, total_lines, week_start
from tests.conftest import FixedRates, requires_db

TODAY: date = date(2026, 11, 4)  # a Wednesday
RATES: Rates = Rates(
    on=date(2026, 10, 31), rub={"RUB": Decimal(1), "VND": Decimal("0.0031"), "USD": Decimal(80)}
)


@pytest.mark.parametrize(
    ("day", "monday"),
    [
        (date(2026, 11, 2), date(2026, 11, 2)),  # Monday itself
        (TODAY, date(2026, 11, 2)),
        (date(2026, 11, 8), date(2026, 11, 2)),  # Sunday
        (date(2026, 11, 1), date(2026, 10, 26)),  # across a month
    ],
)
def test_week_starts_on_monday(day: date, monday: date) -> None:
    assert week_start(day) == monday


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (TODAY, TODAY, "Сегодня, 4 ноября"),
        (date(2026, 11, 3), date(2026, 11, 3), "Вчера, 3 ноября"),
        (date(2026, 10, 28), date(2026, 10, 28), "28 октября"),
        (date(2026, 11, 2), TODAY, "С 2 ноября по 4 ноября"),
    ],
)
def test_title(start: date, end: date, expected: str) -> None:
    assert title(start, end, TODAY) == expected


def test_total_in_dong_and_roubles() -> None:
    lines: list[str] = total_lines({"VND": Decimal(100000), "USD": Decimal(10)}, RATES)
    assert lines == ["Итого: 358 065 ₫ · 1 110 ₽", "Курс ЦБ на 31.10.2026: 10 000 ₫ = 31.00 ₽"]


def test_a_currency_without_a_rate_is_left_out_and_named() -> None:
    lines: list[str] = total_lines({"VND": Decimal(100000), "XYZ": Decimal(5)}, RATES)
    assert lines[0] == "Итого: 100 000 ₫ · 310 ₽"
    assert lines[-1] == "Нет курса, в итог не вошло: 5 XYZ"


def test_without_rates_the_totals_stand() -> None:
    assert total_lines({"USD": Decimal(10), "VND": Decimal(5000)}, None) == [
        "Итого: 10 USD, 5 000 ₫",
        "Курса ЦБ нет: в рублях не посчитал.",
    ]


async def seed(sessions: async_sessionmaker[AsyncSession]) -> list[Expense]:
    session: AsyncSession
    async with sessions() as session:
        return await add_expenses(
            session,
            1,
            [
                NewExpense(amount=Decimal(60000), category=Category.GROCERIES, note="рынок", spent_on=TODAY),
                NewExpense(amount=Decimal(40000), category=Category.GROCERIES, spent_on=date(2026, 11, 3)),
                NewExpense(amount=Decimal(45000), category=Category.TRANSPORT, spent_on=date(2026, 11, 2)),
                NewExpense(amount=Decimal(99000), category=Category.TRANSPORT, spent_on=date(2026, 11, 1)),
            ],
        )


@requires_db
async def test_one_day_lists_expenses(sessions: async_sessionmaker[AsyncSession]) -> None:
    rows: list[Expense] = await seed(sessions)
    result: Report = await report(sessions, FixedRates(), TODAY, TODAY, TODAY)
    assert result.text.splitlines()[:3] == [
        "Сегодня, 4 ноября",
        "1. 60 000 ₫ Продукты «рынок»",
        "Итого: 60 000 ₫ · 189 ₽",
    ]
    assert f"#{rows[0].id}" in result.for_model() and "#" not in result.text


@requires_db
async def test_a_period_goes_by_category(sessions: async_sessionmaker[AsyncSession]) -> None:
    await seed(sessions)
    result: Report = await report(sessions, FixedRates(), TODAY, date(2026, 11, 2), TODAY)
    assert result.rows == [] and result.for_model() == result.text
    assert result.text.splitlines()[:4] == [
        "С 2 ноября по 4 ноября",
        "Продукты: 100 000 ₫ (2)",
        "Транспорт: 45 000 ₫ (1)",  # Sunday's 99 000 is another week
        "Итого: 145 000 ₫ · 456 ₽",
    ]


@requires_db
async def test_an_empty_day(sessions: async_sessionmaker[AsyncSession]) -> None:
    assert (await report(sessions, FixedRates(), TODAY, TODAY, TODAY)).text == "Сегодня, 4 ноября: трат нет"
