"""Rates of the Bank of Russia from a saved answer of theirs; the network is never asked."""

from datetime import date
from decimal import Decimal

import httpx
import pytest

from expense_tracker.rates import BankOfRussia, Rates, UnknownCurrency
from tests.conftest import CBR_DAILY, RATES

MONDAY: date = date(2026, 11, 2)


def test_rates_are_per_one_unit_whatever_the_nominal() -> None:
    assert RATES.on == date(2026, 10, 31)  # Saturday's rate, served on Monday too
    assert RATES.rub["VND"] == Decimal("0.00314223")  # 31,4223 for 10 000 dong
    assert RATES.rub["THB"] == Decimal("2.39035")  # 23,9035 for 10 baht
    assert RATES.rub["RUB"] == Decimal(1)


def test_convert_goes_through_the_rouble() -> None:
    assert RATES.convert(Decimal(1_000_000), "VND", "RUB") == Decimal("3142.23")
    assert round(RATES.convert(Decimal(1), "USD", "VND"), 2) == Decimal("25636.98")
    assert RATES.convert(Decimal(5), "RUB", "RUB") == Decimal(5)


def test_unknown_currency() -> None:
    with pytest.raises(UnknownCurrency, match="XYZ"):
        RATES.convert(Decimal(1), "XYZ", "RUB")


class Bank:
    """Counts requests; answers with the saved XML, or with a 500 while `down`."""

    def __init__(self) -> None:
        self.requests: int = 0
        self.down: bool = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        return httpx.Response(500) if self.down else httpx.Response(200, content=CBR_DAILY)

    def source(self) -> BankOfRussia:
        return BankOfRussia(transport=httpx.MockTransport(self.handle))


async def test_asks_the_bank_once_a_day() -> None:
    bank: Bank = Bank()
    source: BankOfRussia = bank.source()
    first: Rates = await source.latest(MONDAY)
    await source.latest(MONDAY)
    assert bank.requests == 1
    await source.latest(date(2026, 11, 3))
    assert bank.requests == 2
    assert first == RATES


async def test_last_rates_when_the_bank_is_down() -> None:
    bank: Bank = Bank()
    source: BankOfRussia = bank.source()
    await source.latest(MONDAY)
    bank.down = True
    assert await source.latest(date(2026, 11, 3)) == RATES
    assert await source.latest(date(2026, 11, 3)) == RATES
    assert bank.requests == 3  # stale rates are not cached as today's: each call tries again


async def test_no_rates_ever_is_an_error() -> None:
    bank: Bank = Bank()
    bank.down = True
    with pytest.raises(httpx.HTTPStatusError):
        await bank.source().latest(MONDAY)
