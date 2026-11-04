from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from expense_tracker.expenses import (
    Category,
    CategoryTotal,
    ExpenseChanges,
    NewExpense,
    Period,
    add_expenses,
    delete_expense,
    edit_expense,
    list_expenses,
    summary,
)
from expense_tracker.models import Expense
from tests.conftest import requires_db

WIFE: int = 222
ME: int = 111
WEEK: Period = Period(start=date(2026, 11, 2), end=date(2026, 11, 9))


def spent(amount: str, category: Category, day: date, currency: str = "VND") -> NewExpense:
    return NewExpense(amount=Decimal(amount), category=category, spent_on=day, currency=currency)


def test_previous_period_has_the_same_length() -> None:
    assert WEEK.previous() == Period(start=date(2026, 10, 26), end=date(2026, 11, 2))


def test_empty_period_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Period(start=date(2026, 11, 2), end=date(2026, 11, 2))


def test_amount_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        spent("0", Category.GROCERIES, date(2026, 11, 2))


@requires_db
async def test_add_several_at_once(session: AsyncSession) -> None:
    items: list[NewExpense] = [
        NewExpense(
            amount=Decimal(60000), category=Category.EATING_OUT, note="фо", spent_on=date(2026, 11, 2)
        ),
        NewExpense(
            amount=Decimal(35000), category=Category.EATING_OUT, note="кофе", spent_on=date(2026, 11, 2)
        ),
    ]
    rows: list[Expense] = await add_expenses(session, ME, items)
    assert [r.id for r in rows] == sorted(r.id for r in rows)
    assert {r.currency for r in rows} == {"VND"}
    assert [r.note for r in await list_expenses(session, WEEK)] == ["фо", "кофе"]


@requires_db
async def test_database_rejects_non_positive_amount(session: AsyncSession) -> None:
    """The CHECK constraint holds even if someone bypasses NewExpense."""
    session.add(
        Expense(user_id=ME, amount=Decimal(-1), currency="VND", category="other", spent_on=date.today())
    )
    with pytest.raises(IntegrityError):
        await session.flush()


@requires_db
async def test_edit_changes_only_given_fields(session: AsyncSession) -> None:
    row: Expense = (
        await add_expenses(session, ME, [spent("60000", Category.EATING_OUT, date(2026, 11, 2))])
    )[0]
    edited: Expense | None = await edit_expense(session, row.id, ExpenseChanges(amount=Decimal(50000)))
    assert edited is not None
    assert (edited.amount, edited.category) == (Decimal(50000), Category.EATING_OUT)
    assert await edit_expense(session, row.id + 1000, ExpenseChanges(amount=Decimal(1))) is None


@requires_db
async def test_delete(session: AsyncSession) -> None:
    row: Expense = (await add_expenses(session, ME, [spent("10000", Category.OTHER, date(2026, 11, 3))]))[0]
    assert await delete_expense(session, row.id) == date(2026, 11, 3)
    assert await delete_expense(session, row.id) is None
    assert await list_expenses(session, WEEK) == []


@requires_db
async def test_list_filters_by_period_and_category(session: AsyncSession) -> None:
    await add_expenses(
        session,
        ME,
        [
            spent("1000", Category.GROCERIES, date(2026, 11, 1)),  # day before the week
            spent("2000", Category.GROCERIES, date(2026, 11, 2)),  # first day
            spent("3000", Category.TRANSPORT, date(2026, 11, 8)),  # last day
            spent("4000", Category.GROCERIES, date(2026, 11, 9)),  # end is exclusive
        ],
    )
    assert [r.amount for r in await list_expenses(session, WEEK)] == [Decimal(2000), Decimal(3000)]
    assert [r.amount for r in await list_expenses(session, WEEK, Category.GROCERIES)] == [Decimal(2000)]


@requires_db
async def test_summary_compares_with_previous_period(session: AsyncSession) -> None:
    await add_expenses(
        session,
        ME,
        [
            spent("100000", Category.GROCERIES, date(2026, 11, 2)),
            spent("50000", Category.GROCERIES, date(2026, 11, 5)),
            spent("80000", Category.GROCERIES, date(2026, 10, 28)),  # previous week
            spent("30000", Category.TRANSPORT, date(2026, 10, 27)),  # previous week only
            spent("12.50", Category.TRAVEL, date(2026, 11, 4), currency="USD"),
            spent("999999", Category.HOUSING, date(2026, 10, 1)),  # outside both periods
        ],
    )
    await add_expenses(session, WIFE, [spent("20000", Category.EATING_OUT, date(2026, 11, 6))])
    totals: list[CategoryTotal] = await summary(session, WEEK)
    assert totals == [
        CategoryTotal(
            category=Category.TRAVEL,
            currency="USD",
            total=Decimal("12.50"),
            count=1,
            previous_total=Decimal(0),
        ),
        CategoryTotal(
            category=Category.GROCERIES,
            currency="VND",
            total=Decimal(150000),
            count=2,
            previous_total=Decimal(80000),
        ),
        CategoryTotal(
            category=Category.EATING_OUT,
            currency="VND",
            total=Decimal(20000),
            count=1,
            previous_total=Decimal(0),
        ),
        CategoryTotal(
            category=Category.TRANSPORT,
            currency="VND",
            total=Decimal(0),
            count=0,
            previous_total=Decimal(30000),
        ),
    ]
