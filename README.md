# expense-tracker

A family expense tracker in Telegram. Write to the bot the way you would jot it down in a notebook —
«фо 60к», «нет, было 50» — and an LLM agent records, corrects and summarises expenses. Once a week
it drafts a post about how the family lives and publishes it to Threads and a Telegram channel after
your approval.

> **Status:** skeleton. Settings, the bot on long polling with a user whitelist, Postgres, CI.
> The agent, database schema and posting flow are next.

## Design

- **Runs on a laptop at home.** Long polling needs no public IP, domain or VPS; if the laptop sleeps,
  Telegram keeps messages for a day. `docker compose` restarts everything after a reboot.
- **Expense agent** — LangChain `create_agent` with tools; the numbers come from SQL in the tools,
  the model only retells them.
- **Posting flow** — its own LangGraph `StateGraph`: figures → draft → pause for approval (hours,
  surviving a restart) → edit loop → publish to each platform, retrying only the one that failed.
- **PostgreSQL** for expenses and LangGraph checkpoints alike.

## Run

```sh
cp .env.example .env   # bot token, allowed user ids, model key
docker compose up -d
```

Required: `TELEGRAM_BOT_TOKEN`, `ALLOWED_USER_IDS`, `LLM_MODEL` and its provider key. The default
model is `openai/gpt-oss-120b` on Groq's free tier, so it runs without a card; switch to Claude by
changing `LLM_MODEL`. A missing key stops the bot at start with the variable named.

## Develop

Python 3.14, [uv](https://docs.astral.sh/uv/).

```sh
uv sync
docker compose up -d db
DATABASE_URL=postgresql://tracker:tracker@localhost:5432/tracker uv run pytest
uv run ruff check && uv run ruff format --check && uv run mypy
```

Database tests skip without `DATABASE_URL`; CI runs them against a Postgres service. Every
variable, parameter and return value is typed: `mypy --strict` checks signatures,
`tests/test_annotations.py` checks the rest.
