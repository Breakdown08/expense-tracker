"""The eval's judge: a wrong verdict would make the accuracy in the README a lie."""

from datetime import date
from decimal import Decimal

from evals.run import HERE, TODAY, Case, Spent, Verdict, judge, load
from expense_tracker.expenses import Category
from expense_tracker.models import Expense


def row(amount: str, category: str, currency: str = "VND", spent_on: date = TODAY) -> Expense:
    return Expense(amount=Decimal(amount), currency=currency, category=category, spent_on=spent_on)


FO: Spent = Spent(amount=Decimal(60000), category=Category.EATING_OUT)


def test_right_in_any_order() -> None:
    expected: list[Spent] = [FO, Spent(amount=Decimal(9000), category=Category.GROCERIES)]
    got: list[Expense] = [row("9000.00", "groceries"), row("60000.00", "eating_out")]
    assert judge(expected, got) == Verdict(amounts=True, exact=True)


def test_wrong_category_keeps_the_amounts() -> None:
    assert judge([FO], [row("60000", "groceries")]) == Verdict(amounts=True, exact=False)


def test_wrong_day_amount_or_currency_fails_both() -> None:
    wrong: Verdict = Verdict(amounts=False, exact=False)
    assert judge([FO], [row("60000", "eating_out", spent_on=date(2026, 11, 1))]) == wrong
    assert judge([FO], [row("6000", "eating_out")]) == wrong
    assert judge([FO], [row("60000", "eating_out", currency="USD")]) == wrong


def test_recording_nothing_or_twice_fails() -> None:
    wrong: Verdict = Verdict(amounts=False, exact=False)
    assert judge([FO], []) == wrong
    assert judge([FO], [row("60000", "eating_out"), row("60000", "eating_out")]) == wrong
    assert judge([], [row("60000", "eating_out")]) == wrong


def test_nothing_expected_nothing_recorded() -> None:
    assert judge([], []) == Verdict(amounts=True, exact=True)


def test_phrases_load() -> None:
    cases: list[Case] = load(HERE / "phrases.jsonl")
    assert len(cases) >= 50
