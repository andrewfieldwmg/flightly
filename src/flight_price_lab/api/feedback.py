"""Capture explicit shared preferences without retaining routes or travel dates."""

import hashlib
import json
import os
import secrets
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from flight_price_lab.api.models import SearchSnapshot, TripOption
from flight_price_lab.storage.database import SearchSessionEntry
from flight_price_lab.storage.preference_feedback import (
    PreferenceFeedbackEntry,
    PreferenceFeedbackStore,
    digest_request,
)


class SharedChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: UUID
    participant_id: UUID
    consent: Literal[True]
    search_id: str = Field(min_length=1, max_length=100)
    chosen_id: str = Field(min_length=1, max_length=256)
    rejected_id: str = Field(min_length=1, max_length=256)
    model_version: str = Field(pattern=r"^\d{1,12}$")
    model_name: Literal[
        "FlightlyPreferenceRanker", "FlightlySharedPreferenceRanker"
    ] = "FlightlyPreferenceRanker"
    profile_version: Literal[1]


def learning_input(option: TripOption) -> dict:
    fields = {
        "base_price",
        "total_journey_minutes",
        "is_nonstop",
        "is_self_transfer",
        "connection_minutes",
        "ticketing_type",
        "baggage_confidence",
        "price_completeness",
    }
    result = option.model_dump(mode="json", include=fields)
    # The ranker uses local clock time. Remove the actual date and route.
    result["departure_at"] = f"2000-01-01T{option.departure_at:%H:%M}:00Z"
    result["arrival_at"] = f"2000-01-01T{option.arrival_at:%H:%M}:00Z"
    return result


def feedback_router(engine: Engine) -> APIRouter:
    router = APIRouter(prefix="/api/learning")
    store = PreferenceFeedbackStore(engine)

    @router.get("/export", include_in_schema=False)
    def export_feedback(
        response: Response,
        authorization: str = Header(default=""),
        after: str = Query(default="", max_length=36),
    ) -> dict:
        response.headers["Cache-Control"] = "no-store"
        token = os.getenv("FLIGHTLY_FEEDBACK_SYNC_TOKEN", "")
        if len(token) < 32 or not secrets.compare_digest(
            authorization, f"Bearer {token}"
        ):
            raise HTTPException(401, "Private feedback sync authorization required")
        with Session(engine) as session:
            rows = session.scalars(
                select(PreferenceFeedbackEntry)
                .where(PreferenceFeedbackEntry.event_id > after)
                .order_by(PreferenceFeedbackEntry.event_id)
                .limit(500)
            ).all()
            records = [
                dict(
                    json.loads(row.payload_json),
                    event_id=row.event_id,
                    participant=row.participant_hash,
                    model_version=row.model_version,
                    created_at=row.created_at.isoformat(),
                )
                for row in rows
            ]
        return {
            "records": records,
            "next": rows[-1].event_id if len(rows) == 500 else None,
        }

    @router.post("/feedback", status_code=201)
    def feedback(choice: SharedChoice) -> dict:
        digest = digest_request(choice.model_dump(mode="json"))
        try:
            if store.existing(str(choice.event_id), digest):
                return {"event_id": str(choice.event_id), "duplicate": True}
            with Session(engine) as session:
                entry = session.get(SearchSessionEntry, choice.search_id)
                if entry is None:
                    raise HTTPException(404, "The original search is unavailable")
                snapshot = SearchSnapshot.model_validate_json(entry.snapshot_json)
            groups = [snapshot.outbound, snapshot.return_]
            for group in filter(None, groups):
                candidates = [
                    group.baseline,
                    *group.nonstop_options,
                    *group.feasible_options,
                ]
                options = {
                    option.id: option for option in candidates if option is not None
                }
                if choice.chosen_id in options and choice.rejected_id in options:
                    break
            else:
                raise HTTPException(
                    422, "Choose two options from the same search direction"
                )
            if choice.chosen_id == choice.rejected_id:
                raise HTTPException(422, "The two options must be different")
            payload = {
                "chosen": learning_input(options[choice.chosen_id]),
                "rejected": learning_input(options[choice.rejected_id]),
                "origin": "human",
                "model_name": choice.model_name,
            }
            inserted = store.add(
                {
                    "event_id": str(choice.event_id),
                    "participant_hash": hashlib.sha256(
                        str(choice.participant_id).encode()
                    ).hexdigest(),
                    "request_digest": digest,
                    "model_version": choice.model_version,
                    "profile_version": choice.profile_version,
                    "payload_json": json.dumps(payload),
                    "created_at": datetime.now(UTC),
                }
            )
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except OverflowError as error:
            raise HTTPException(429, str(error)) from error
        return {"event_id": str(choice.event_id), "duplicate": not inserted}

    return router
