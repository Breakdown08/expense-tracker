# expense-tracker

A family expense tracker in Telegram. Write to the bot the way you would jot it down in a notebook —
«фо 60к», «нет, было 50» — and an LLM agent records, corrects and summarises expenses. Once a week
it drafts a post about how the family lives and publishes it to Threads and a Telegram channel after
your approval.

> **Status:** the bot records, corrects and summarises expenses through the agent; memory per chat
> in Postgres. The posting flow and the model eval are next.

## Design

- **Runs on a laptop at home.** Long polling needs no public IP, domain or VPS; if the laptop sleeps,
  Telegram keeps messages for a day. `docker compose` restarts everything after a reboot.
- **Expense agent** — LangChain `create_agent` with tools over `expenses.py`; the numbers come
  from SQL in the tools, the model only retells them. Middleware instead of hand-written loops:
  history folds into a summary past 3K tokens (Groq's free tier allows 8K a minute), rate limits
  and provider 5xx are retried while a wrong key is not, and one message is capped at 6 model
  calls. The chat id is the LangGraph thread, kept by the Postgres checkpointer across restarts.
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

The bot works in a private chat or in a group shared by the family: there everyone sees every
record, and the agent keeps one history for the group, each message tagged with its author. For a
group, turn privacy mode off in @BotFather (`/setprivacy` → Disable) *before* adding the bot,
otherwise it sees only commands; afterwards `/setjoingroups` → Disable keeps it out of other groups.

## Develop

Python 3.14, [uv](https://docs.astral.sh/uv/).

```sh
uv sync
docker compose up -d db
DATABASE_URL=postgresql://tracker:tracker@localhost:5432/tracker uv run pytest
uv run ruff check && uv run ruff format --check && uv run ty check && uv run mypy
```

Database tests skip without `DATABASE_URL`; CI runs them against a Postgres service. They create
their own `<name>_test` database next to it, so your data stays untouched.

The bot applies migrations at start. After changing `models.py`:

```sh
uv run alembic revision --autogenerate -m "what changed"
```

`tests/test_migrations.py` fails if a model changed without a revision.

Agent tests run on a scripted model: they check tools, memory, limits and retries, not how well a
real model understands «фо 60к» — that is the eval's job. Every
variable, parameter and return value is typed: ty and `mypy --strict` check the code,
`tests/test_annotations.py` checks that variables are annotated.
