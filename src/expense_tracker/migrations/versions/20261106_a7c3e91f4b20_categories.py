"""the family's own categories

Revision ID: a7c3e91f4b20
Revises: dad089501c2e
Create Date: 2026-11-06 12:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7c3e91f4b20"
down_revision: str | Sequence[str] | None = "dad089501c2e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UP: tuple[tuple[str, str], ...] = (
    ("coffee", r"category = 'eating_out' AND note ~* 'коф|coffee'"),
    ("drinks", r"category IN ('eating_out', 'groceries') AND note ~* '\m(напит|чай|чая|сок)'"),
    ("mobile", r"category = 'subscriptions' AND note ~* 'интернет|связ|тариф'"),
    ("cafe", "category = 'eating_out'"),
    ("taxi", "category = 'transport'"),
    ("rent", "category = 'housing'"),
    ("pharmacy", "category = 'health'"),
)

DOWN: tuple[tuple[str, str], ...] = (
    ("eating_out", "category IN ('coffee', 'cafe')"),
    ("groceries", "category = 'drinks'"),
    ("subscriptions", "category = 'mobile'"),
    ("transport", "category = 'taxi'"),
    ("housing", "category = 'rent'"),
    ("health", "category = 'pharmacy'"),
)


def recategorize(rules: tuple[tuple[str, str], ...]) -> None:
    category: str
    where: str
    for category, where in rules:
        op.get_bind().execute(
            sa.text(f"UPDATE expenses SET category = :category WHERE {where}"), {"category": category}
        )


def upgrade() -> None:
    left: int | None = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT count(*) FROM expenses WHERE category IN ('clothes', 'leisure', 'travel', 'other')"
            )
        )
        .scalar()
    )
    if left:
        raise RuntimeError(f"{left} expenses have no new category: move them by hand first")
    recategorize(UP)


def downgrade() -> None:
    recategorize(DOWN)
