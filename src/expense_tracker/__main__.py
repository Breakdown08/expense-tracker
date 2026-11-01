"""`python -m expense_tracker` — what the container runs."""

import asyncio
import logging
import sys

from pydantic import ValidationError

from expense_tracker.bot import run
from expense_tracker.config import Settings, explain


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings: Settings
    try:
        settings = Settings()
    except ValidationError as error:
        sys.exit(explain(error))
    asyncio.run(run(settings))


if __name__ == "__main__":
    main()
