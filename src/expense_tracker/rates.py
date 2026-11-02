"""Exchange rates of the Bank of Russia: the rouble is the bridge between any two currencies.

Past expenses convert at the latest rate, not at the rate of their day (Kirill, 2026-11-02):
two weeks side by side then differ by what was spent, not by how the rate moved.
"""

import asyncio
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Protocol, runtime_checkable

import httpx

log: logging.Logger = logging.getLogger(__name__)

# Daily rates for every currency the Bank sets, VND included (per 10 000 dong); no key needed.
CBR_DAILY: str = "https://www.cbr.ru/scripts/XML_daily.asp"


class UnknownCurrency(LookupError):
    pass


@dataclass(frozen=True)
class Rates:
    on: date  # the Bank's date, not ours: Saturday's rate also serves Sunday and Monday
    rub: dict[str, Decimal]  # roubles for one unit of each currency, RUB itself included

    def convert(self, amount: Decimal, currency: str, to: str) -> Decimal:
        code: str
        for code in (currency, to):
            if code not in self.rub:
                raise UnknownCurrency(code)
        return amount * self.rub[currency] / self.rub[to]


def field(valute: ET.Element, tag: str) -> str:
    value: str | None = valute.findtext(tag)
    if value is None:
        raise ValueError(f"<Valute> without <{tag}>")
    return value


def parse(document: bytes) -> Rates:
    """XML_daily.asp: windows-1251, decimal commas, a rate per Nominal units (10 000 VND, 10 THB)."""
    root: ET.Element = ET.fromstring(document)
    rub: dict[str, Decimal] = {"RUB": Decimal(1)}
    valute: ET.Element
    for valute in root.iter("Valute"):
        value: Decimal = Decimal(field(valute, "Value").replace(",", "."))
        rub[field(valute, "CharCode")] = value / Decimal(field(valute, "Nominal"))
    return Rates(on=datetime.strptime(root.attrib["Date"], "%d.%m.%Y").date(), rub=rub)


# runtime_checkable: LangChain builds a pydantic schema of the tools' Context, which isinstance-checks it.
@runtime_checkable
class RateSource(Protocol):
    async def latest(self, today: date) -> Rates: ...


class BankOfRussia:
    """Asks the Bank once a family day. If it does not answer, the last rates beat none."""

    def __init__(
        self, url: str = CBR_DAILY, timeout: float = 10.0, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.url: str = url
        self.timeout: float = timeout
        self.transport: httpx.AsyncBaseTransport | None = transport  # tests answer without the network
        self.cached: Rates | None = None
        self.fetched_on: date | None = None
        self.lock: asyncio.Lock = asyncio.Lock()  # two chats asking at once make one request

    async def latest(self, today: date) -> Rates:
        async with self.lock:
            if self.cached is not None and self.fetched_on == today:
                return self.cached
            try:
                self.cached = parse(await self.fetch())
                self.fetched_on = today
            except httpx.HTTPError, ET.ParseError, ValueError, KeyError, InvalidOperation:
                if self.cached is None:
                    raise
                log.warning("Bank of Russia unavailable, using rates of %s", self.cached.on, exc_info=True)
            return self.cached

    async def fetch(self) -> bytes:
        client: httpx.AsyncClient
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            response: httpx.Response = await client.get(self.url)
            response.raise_for_status()
            return response.content
