"""Recording and reading expenses. The numbers come from SQL here; the model only retells them."""

from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import Row, Select, TextClause, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from expense_tracker.models import Expense


class Category(StrEnum):
    GROCERIES = "groceries"
    EATING_OUT = "eating_out"
    TRANSPORT = "transport"
    HOUSING = "housing"  # rent, utilities
    SUBSCRIPTIONS = "subscriptions"  # internet, phone plan, streaming, apps
    HOUSEHOLD = "household"
    HEALTH = "health"
    CLOTHES = "clothes"
    LEISURE = "leisure"
    TRAVEL = "travel"
    OTHER = "other"


class NewExpense(BaseModel):
    amount: Decimal = Field(gt=0, max_digits=14, decimal_places=2)
    currency: str = Field(default="VND", pattern=r"^[A-Z]{3}$")
    category: Category
    note: str | None = None
    spent_on: date


class ExpenseChanges(BaseModel):
    """Only the fields that are set change."""

    amount: Decimal | None = Field(default=None, gt=0, max_digits=14, decimal_places=2)
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    category: Category | None = None
    note: str | None = None
    spent_on: date | None = None


class Period(BaseModel):
    """Days from `start` up to, not including, `end`."""

    start: date
    end: date

    @model_validator(mode="after")
    def not_empty(self) -> Self:
        if self.start >= self.end:
            raise ValueError(f"period is empty: {self.start} .. {self.end}")
        return self

    def previous(self) -> Period:
        """The same number of days right before: a week before a week, 30 days before 30 days."""
        length: timedelta = self.end - self.start
        return Period(start=self.start - length, end=self.start)


class CategoryTotal(BaseModel):
    category: Category
    currency: str
    total: Decimal
    count: int
    previous_total: Decimal  # same category and currency over Period.previous()


async def add_expenses(session: AsyncSession, user_id: int, items: list[NewExpense]) -> list[Expense]:
    """Several at once: the model makes one tool call for «фо 60к, кофе 35к»."""
    rows: list[Expense] = [Expense(user_id=user_id, **item.model_dump()) for item in items]
    session.add_all(rows)
    await session.commit()
    return rows


async def edit_expense(session: AsyncSession, expense_id: int, changes: ExpenseChanges) -> Expense | None:
    row: Expense | None = await session.get(Expense, expense_id)
    if row is None:
        return None
    field: str
    value: object
    for field, value in changes.model_dump(exclude_unset=True).items():
        setattr(row, field, value)
    await session.commit()
    return row


async def delete_expense(session: AsyncSession, expense_id: int) -> bool:
    row: Expense | None = await session.get(Expense, expense_id)
    if row is None:
        return False
    await session.delete(row)
    await session.commit()
    return True


async def list_expenses(
    session: AsyncSession, period: Period, category: Category | None = None
) -> list[Expense]:
    query: Select[Expense] = (
        select(Expense)
        .where(Expense.spent_on >= period.start, Expense.spent_on < period.end)
        .order_by(Expense.spent_on, Expense.id)
    )
    if category is not None:
        query = query.where(Expense.category == category)
    return list((await session.scalars(query)).all())


# One pass over both periods: FILTER splits the sums, a category seen only last time still shows up.
SUMMARY: TextClause = text("""
    SELECT category,
           currency,
           coalesce(sum(amount) FILTER (WHERE spent_on >= :start), 0) AS total,
           count(*) FILTER (WHERE spent_on >= :start)                 AS count,
           coalesce(sum(amount) FILTER (WHERE spent_on < :start), 0)  AS previous_total
    FROM expenses
    WHERE spent_on >= :previous_start AND spent_on < :end
    GROUP BY category, currency
    ORDER BY currency, total DESC, category
""")


async def summary(session: AsyncSession, period: Period) -> list[CategoryTotal]:
    """Totals per category and currency, next to the previous period of the same length."""
    previous: Period = period.previous()
    rows: list[Row[tuple[str, str, Decimal, int, Decimal]]] = list(
        (
            await session.execute(
                SUMMARY, {"start": period.start, "end": period.end, "previous_start": previous.start}
            )
        ).all()
    )
    return [CategoryTotal.model_validate(row._asdict()) for row in rows]
