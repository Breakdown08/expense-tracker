"""Engine, sessions and migrations. One driver, psycopg 3, for SQLAlchemy and LangGraph alike."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

MIGRATIONS: Path = Path(__file__).parent / "migrations"


def sqlalchemy_url(url: str) -> str:
    """DATABASE_URL is libpq-style (`postgresql://`), as psycopg and LangGraph take it."""
    return url.replace("postgresql://", "postgresql+psycopg://", 1)


def alembic_config(url: str) -> Config:
    config: Config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.attributes["url"] = url
    return config


def upgrade(url: str) -> None:
    """Bring the schema to the latest revision; also fails early when Postgres is unreachable."""
    command.upgrade(alembic_config(url), "head")


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(sqlalchemy_url(url))


def make_sessions(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
