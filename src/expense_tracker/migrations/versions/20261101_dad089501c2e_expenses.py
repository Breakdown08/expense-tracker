"""expenses

Revision ID: dad089501c2e
Revises:
Create Date: 2026-11-01 17:28:21.656376
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "dad089501c2e"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "expenses",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("amount", sa.Numeric(precision=14, scale=2), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("spent_on", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("amount > 0", name=op.f("ck_expenses_amount_positive")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_expenses")),
    )
    op.create_index(op.f("ix_expenses_spent_on"), "expenses", ["spent_on"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_expenses_spent_on"), table_name="expenses")
    op.drop_table("expenses")
