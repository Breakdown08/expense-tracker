"""`python -m expense_tracker` — what the container runs."""

import asyncio
import logging
import os
import sys

from pydantic import ValidationError

from expense_tracker.bot import run
from expense_tracker.config import DEFAULT_DATABASE_URL, Settings, explain
from expense_tracker.db import upgrade

USAGE: str = "usage: expense-tracker [migrate]"


def main(argv: list[str] | None = None) -> None:
    args: list[str] = sys.argv[1:] if argv is None else argv
    if args not in ([], ["migrate"]):
        print(USAGE, file=sys.stderr)
        sys.exit(2)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args == ["migrate"]:  # for the Kubernetes initContainer: needs the database only, not the bot config
        upgrade(os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
        logging.getLogger(__name__).info("migrations applied")
        return
    settings: Settings
    try:
        settings = Settings()
    except ValidationError as error:
        sys.exit(explain(error))
    upgrade(settings.database_url)  # before the bot answers anyone: the schema is current
    asyncio.run(run(settings))


if __name__ == "__main__":
    main()
