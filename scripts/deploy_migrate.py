"""Run additive Alembic migrations in production's build environment, before release."""

import os
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flight_price_lab.storage.database import create_database_engine

if os.getenv("VERCEL_ENV") == "production":
    engine = create_database_engine()
    with engine.begin() as connection:
        # Parallel deployments must not race the schema-version update.
        connection.execute(text("SELECT pg_advisory_xact_lock(1863812541)"))
        config = Config("alembic.ini")
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    engine.dispose()
else:
    print("Skipping production schema migration outside a production build")
