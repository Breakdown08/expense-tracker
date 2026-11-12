# expense-tracker

A family expense tracker in Telegram. Write to the bot the way you would jot it down in a notebook —
«фо 60к», «нет, было 50» — and an LLM agent records, corrects and summarises expenses. On `/отчёт` it
drafts a post about how the family's month went and publishes it to Threads and a Telegram channel
after your approval.

> **Status:** in daily use since November 2026 for recording and summaries. The posting flow works
> up to the draft on a real model and is tested against recorded platforms; publishing to Threads
> and to a channel has not been run live yet.

## Design

- **Runs on a laptop at home.** Long polling needs no public IP, domain or VPS; if the laptop sleeps,
  Telegram keeps messages for a day. The bot runs in a local k3d cluster (see Kubernetes below), `docker compose` is the fallback.
- **Expense agent** — LangChain `create_agent` with tools over `expenses.py`; the numbers come
  from SQL in the tools, the model only retells them. Middleware instead of hand-written loops:
  rate limits and provider 5xx are retried while a wrong key is not, and one message is capped at
  6 model calls. Memory is a LangGraph thread per chat **per day**, kept by the Postgres
  checkpointer across restarts: «нет, было 50» finds today's record, while last month comes from
  SQL, not from the chat. A year of use costs no more tokens per message than the first day.
  Tools report totals too, counted in SQL: the model retells them and never adds up.
- **Code before the model.** `/сегодня`, `/вчера`, `/неделя` are plain templates over SQL and never
  reach the agent. A question in words reaches it, but when a tool already holds the answer — a
  record saved, a day's report, a conversion — the graph ends right after the tool, so the model
  reads the text and picks the tool, and does not retell the result. That halved the tokens: the eval
  went from 5 400 to 2 900 input tokens a case, same score. Comparing periods still goes back to the
  model. It is a `before_model` hook, not `return_direct`, which would send a tool error to the chat.
- **Posting flow** — its own LangGraph `StateGraph` (`post.py`), since a run of four calls could not
  do this: figures for the last 30 days (SQL: amounts in dong, shares, changes) → draft (LLM) → pause for
  approval, for hours and across restarts → edit loop → one platform per step, so a failure or a
  crash never posts twice → retry only what failed. Without platform keys the bot leaves the text to
  copy by hand.
- **PostgreSQL** for expenses and LangGraph checkpoints alike.

## Measured

`uv run python -m evals.run` sends phrases through the real agent and compares what landed in the
database with what should have: amount, currency, day, category. Questions run against a prepared
book, and their answers must hold the figures worked out by hand. It costs money, so it runs by hand.

| Model | Exact | Amounts | Answers | Tokens per case | Cost of a run |
|---|---|---|---|---|---|
| `claude-haiku-5-5` | 70/70 | 70/70 | 70/70 | 3 400 in, 178 out | $0.03 |

Read that as «no misses on this set», not as 100% accuracy. The first run of the questions did miss:
«сколько вчера вышло в долларах?» got a total in dong, and the prompt was fixed. The 53 phrases
([`evals/phrases.jsonl`](evals/phrases.jsonl)) are written by hand in the style of real messages —
real ones stay private — by the author of the prompt, and they are easy for the model: categories,
foreign currencies, relative dates, several expenses at once, corrections, and messages where it
must ask instead of guessing. The 17 cases in [`evals/questions.jsonl`](evals/questions.jsonl) ask
for totals, periods and conversions, and correct records by their number in a list.
After the move to the family's own ten categories the first run wrote plane tickets into taxi; the
prompt now limits it to rides. Expenses that fit no category must be asked about, not recorded.
The default Groq model has not been run: no key here yet. Raw answers are in
[`evals/results/`](evals/results/).

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

## Kubernetes (k3d)

The bot and Postgres run in a local [k3d](https://k3d.io) cluster on the same laptop; manifests are
in [`k8s/`](k8s/) (Kustomize: `k8s/base` plus the overlay `k8s/overlays/k3d`), `docker compose` stays
as the fallback. Needs docker, k3d and kubectl.

```sh
scripts/k3d-cluster.sh   # create or start the cluster; storage is under $EXPENSE_TRACKER_DATA/k3d-storage
scripts/deploy.sh        # build the image, create the Secret from .env, apply the manifests
```

The Secret `expense-tracker-env` is made from `.env` (`ENV_FILE`) and is never in git. If `.env` has
no `POSTGRES_PASSWORD`, `tracker` is used, as in compose. Changing it later does not change the
password of an already initialised database (same as in compose); it must be usable inside a URL.
`deploy.sh` refuses to run while the compose bot is up: two long pollings with one token conflict.
Variables: `CLUSTER` (default `expense-tracker`), `NAMESPACE`, `EXPENSE_TRACKER_DATA`
(default `~/.local/share/expense-tracker`), `ENV_FILE`, `WAIT_BOT`.

Migrations run in the initContainer `python -m expense_tracker migrate`, which needs only `DATABASE_URL`.

### Backups

A CronJob makes `pg_dump` every day at 22:00 (`Asia/Ho_Chi_Minh`) and keeps the last 14 copies.
Copies lie on the same disk as the database: they save from a dropped table, not from a dead SSD.
On the host they are in `$EXPENSE_TRACKER_DATA/k3d-storage/pvc-*_expense-tracker_backups/`.

```sh
scripts/backup.sh               # a backup now
scripts/restore.sh latest       # restore from the newest copy in the cluster
scripts/restore.sh ~/some.dump  # or from a file on the host
```

A restore stops the bot, replaces the data, and starts the bot again. After `k3d cluster delete`
the new cluster is empty: fill it with `restore.sh` from a copy on the host.

### Moving from compose

```sh
scripts/k3d-cluster.sh
docker compose stop bot
docker compose exec -T db pg_dump -U tracker --format=custom tracker > ~/expense-tracker-compose.dump
scripts/deploy.sh
scripts/restore.sh ~/expense-tracker-compose.dump
```

Compare both sides, the results must match:

```sh
docker compose exec -T db psql -U tracker -d tracker -tAc 'select count(*), sum(amount) from expenses'
kubectl --context k3d-expense-tracker -n expense-tracker exec postgres-0 -- \
  psql -U tracker -d tracker -tAc 'select count(*), sum(amount) from expenses'
```

Then `docker compose stop db` (a manually stopped container does not come back after a reboot).
Keep the volume `pgdata`: it is the way back.

### Back to compose

```sh
kubectl --context k3d-expense-tracker -n expense-tracker scale deployment/bot --replicas=0
kubectl --context k3d-expense-tracker -n expense-tracker exec postgres-0 -- \
  pg_dump -U tracker --format=custom tracker > ~/expense-tracker-k3d.dump
docker compose up -d db
docker compose exec -T db pg_restore --clean --if-exists --single-transaction --no-owner \
  -U tracker -d tracker < ~/expense-tracker-k3d.dump
docker compose up -d
k3d cluster stop expense-tracker   # optional
```

### Shell scripts

CI lints them with shellcheck; on NixOS: `nix shell nixpkgs#shellcheck`.

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
real model understands «фо 60к» — that is [the eval's](#measured) job. Every
variable, parameter and return value is typed: ty and `mypy --strict` check the code,
`tests/test_annotations.py` checks that variables are annotated.
