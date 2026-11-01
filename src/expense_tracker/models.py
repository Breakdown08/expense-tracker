"""Tables. Schema changes go through Alembic: `uv run alembic revision --autogenerate -m ...`."""

from datetime import date, datetime
from decimal import Decimal
from typing import ClassVar

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Identity,
    MetaData,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Stable constraint names, so that migrations can drop what they created.
NAMING: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata: ClassVar[MetaData] = MetaData(naming_convention=NAMING)


class Expense(Base):
    __tablename__: str = "expenses"
    __table_args__: tuple[CheckConstraint, ...] = (CheckConstraint("amount > 0", name="amount_positive"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger)  # Telegram id of who spent
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    currency: Mapped[str] = mapped_column(String(3))
    category: Mapped[str] = mapped_column(String(32))
    note: Mapped[str | None] = mapped_column(Text)
    spent_on: Mapped[date] = mapped_column(Date, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
