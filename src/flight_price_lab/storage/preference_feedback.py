"""Consent-based pairwise labels, derived from stored search options."""

import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, Text, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from flight_price_lab.storage.database import Base


class PreferenceFeedbackEntry(Base):
    __tablename__ = "preference_feedback"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    participant_hash: Mapped[str] = mapped_column(String(64), index=True)
    request_digest: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(24))
    profile_version: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class PreferenceFeedbackStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def existing(self, event_id: str, digest: str) -> bool:
        with Session(self.engine) as session:
            row = session.get(PreferenceFeedbackEntry, event_id)
            if row is None:
                return False
            if row.request_digest != digest:
                raise ValueError("Feedback event ID already belongs to another choice")
            return True

    def add(self, values: dict) -> bool:
        with Session(self.engine) as session, session.begin():
            # Serialise a participant's daily quota check across function instances.
            lock = int(values["participant_hash"][:15], 16)
            session.execute(text("SELECT pg_advisory_xact_lock(:lock)"), {"lock": lock})
            row = session.get(PreferenceFeedbackEntry, values["event_id"])
            if row is not None:
                if row.request_digest != values["request_digest"]:
                    raise ValueError(
                        "Feedback event ID already belongs to another choice"
                    )
                return False
            today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            count = session.scalar(
                select(func.count())
                .select_from(PreferenceFeedbackEntry)
                .where(
                    PreferenceFeedbackEntry.participant_hash
                    == values["participant_hash"],
                    PreferenceFeedbackEntry.created_at >= today,
                )
            )
            if count >= 200:
                raise OverflowError("Daily shared-choice limit reached")
            inserted = session.scalar(
                insert(PreferenceFeedbackEntry)
                .values(**values)
                .on_conflict_do_nothing(index_elements=["event_id"])
                .returning(PreferenceFeedbackEntry.event_id)
            )
            if inserted is None:
                row = session.get(PreferenceFeedbackEntry, values["event_id"])
                if row.request_digest != values["request_digest"]:
                    raise ValueError(
                        "Feedback event ID already belongs to another choice"
                    )
            return inserted is not None


def digest_request(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
