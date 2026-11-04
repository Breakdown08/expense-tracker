"""Answers made by code from a template: commands and direct tool answers spend no tokens."""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from expense_tracker import expenses
from expense_tracker.expenses import Category, CategoryTotal, Period
from expense_tracker.models import Expense
from expense_tracker.rates import Rates, RateSource

log: logging.Logger = logging.getLogger(__name__)

ONE: Decimal = Decimal(1)  # quantize to whole units: round() would return an int

SIGNS: dict[str, str] = {"VND": "₫", "RUB": "₽"}

CATEGORY_NAMES: dict[Category, str] = {
    Category.GROCERIES: "Продукты",
    Category.EATING_OUT: "Кафе",
    Category.TRANSPORT: "Транспорт",
    Category.HOUSING: "Жильё",
    Category.SUBSCRIPTIONS: "Подписки",
    Category.HOUSEHOLD: "Дом",
    Category.HEALTH: "Здоровье",
    Category.CLOTHES: "Одежда",
    Category.LEISURE: "Досуг",
    Category.TRAVEL: "Поездки",
    Category.OTHER: "Другое",
}

MONTHS: tuple[str, ...] = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)


def money(amount: Decimal) -> str:
    """60 000, 12.50: spaces between thousands, cents only when there are any."""
    places: str = ",.0f" if amount == amount.to_integral_value() else ",.2f"
    return format(amount, places).replace(",", " ")


def price(amount: Decimal, currency: str) -> str:
    return f"{money(amount)} {SIGNS.get(currency, currency)}"


def prices(totals: dict[str, Decimal]) -> str:
    return ", ".join(price(total, currency) for currency, total in sorted(totals.items())) or "0"


def day_name(day: date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]}"


def entry(row: Expense) -> str:
    note: str = f" «{row.note}»" if row.note else ""
    return f"{price(row.amount, row.currency)} {CATEGORY_NAMES[Category(row.category)]}{note}"


def with_ids(text: str, rows: list[Expense]) -> str:
    """People read amounts, not keys; the model needs the keys to correct a record."""
    return "\n".join([text, f"Record ids, in the order above: {', '.join(f'#{r.id}' for r in rows)}"])


@dataclass(frozen=True)
class Report:
    text: str
    rows: list[Expense]  # the numbered ones: a period goes by category and has none

    def for_model(self) -> str:
        return with_ids(self.text, self.rows) if self.rows else self.text


def week_start(today: date) -> date:
    """The Monday of this week; on a Monday, today."""
    return today - timedelta(days=today.weekday())


def period(start: date, end: date) -> Period:
    """Both days included, as people name them; Period ends the day after."""
    return Period(start=start, end=end + timedelta(days=1))


def title(start: date, end: date, today: date) -> str:
    if start != end:
        return f"С {day_name(start)} по {day_name(end)}"
    if start == today:
        return f"Сегодня, {day_name(start)}"
    if start == today - timedelta(days=1):
        return f"Вчера, {day_name(start)}"
    return day_name(start).capitalize()


async def latest_rates(rates: RateSource, today: date) -> Rates | None:
    """Without rates the totals per currency still stand."""
    try:
        return await rates.latest(today)
    except Exception:
        log.warning("no exchange rates", exc_info=True)
        return None


def total_lines(totals: dict[str, Decimal], rates: Rates | None) -> list[str]:
    """The total in dong and roubles at the latest rate, as past expenses always convert."""
    known: list[str] = [c for c in totals if rates is not None and c in rates.rub]
    if rates is None or not known:
        return [f"Итого: {prices(totals)}", "Курса ЦБ нет: в рублях не посчитал."]
    whole: list[str] = [
        price(sum((rates.convert(totals[c], c, target) for c in known), Decimal(0)).quantize(ONE), target)
        for target in ("VND", "RUB")
    ]
    rate: Decimal = rates.convert(Decimal(10000), "VND", "RUB")
    lines: list[str] = [
        f"Итого: {whole[0]} · {whole[1]}",
        f"Курс ЦБ на {rates.on:%d.%m.%Y}: 10 000 ₫ = {rate:.2f} ₽",
    ]
    unknown: list[str] = [c for c in totals if c not in known]
    if unknown:
        lines.append(f"Нет курса, в итог не вошло: {prices({c: totals[c] for c in unknown})}")
    return lines


async def report(
    sessions: async_sessionmaker[AsyncSession], rates: RateSource, today: date, start: date, end: date
) -> Report:
    """One day expense by expense, numbered from 1, a longer period by category; start and end are both
    included."""
    head: str = title(start, end, today)
    days: Period = period(start, end)
    session: AsyncSession
    async with sessions() as session:
        totals: list[CategoryTotal] = [t for t in await expenses.summary(session, days) if t.total > 0]
        rows: list[Expense] = await expenses.list_expenses(session, days) if start == end else []
    if not totals:
        return Report(f"{head}: трат нет", [])
    body: list[str] = [f"{n}. {entry(row)}" for n, row in enumerate(rows, 1)] or [
        f"{CATEGORY_NAMES[t.category]}: {price(t.total, t.currency)} ({t.count})" for t in totals
    ]
    spent: dict[str, Decimal] = {}
    t: CategoryTotal
    for t in totals:
        spent[t.currency] = spent.get(t.currency, Decimal(0)) + t.total
    lines: list[str] = [head, *body, *total_lines(spent, await latest_rates(rates, today))]
    return Report("\n".join(lines), rows)
