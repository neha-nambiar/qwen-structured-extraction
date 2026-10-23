"""Alembic entry point: reads DATABASE_URL through Settings and targets the ORM schema."""

from alembic import context
from sqlalchemy import create_engine

from eroll.service.db import Base
from eroll.settings import load_settings, require

settings = load_settings()
require(settings, "database_url")
database_url = settings.database_url.get_secret_value()

if context.is_offline_mode():
    context.configure(url=database_url, target_metadata=Base.metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(database_url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()
