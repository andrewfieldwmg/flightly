import json
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from flight_price_lab.api.feedback import feedback_router
from flight_price_lab.api.models import (
    DirectionResults,
    SearchSnapshot,
    SearchStatus,
    TripOption,
    TripSearchRequest,
)
from flight_price_lab.storage.database import SearchSessionStore, create_database_engine
from flight_price_lab.storage.preference_feedback import PreferenceFeedbackEntry


@pytest.fixture
def stored_pair():
    # Saved options, not a live provider request. Routes/dates must not reach learning storage.
    engine = create_database_engine()
    first = TripOption(
        id="a",
        direction="OUTBOUND",
        route=["STN", "CAG"],
        flight_numbers=["FR 1"],
        airlines=["Ryanair"],
        legs=[],
        base_price=Decimal(150),
        ancillary_price_low=Decimal(0),
        ancillary_price_high=Decimal(0),
        effective_price_low=Decimal(150),
        effective_price_high=Decimal(150),
        price_completeness="COMPLETE",
        ticketing_type="single_ticket",
        baggage_confidence="high",
        currency="GBP",
        departure_at=datetime.fromisoformat("2026-12-18T11:00:00+00:00"),
        arrival_at=datetime.fromisoformat("2026-12-18T13:30:00+00:00"),
        total_journey_minutes=150,
        is_nonstop=True,
        is_self_transfer=False,
    )
    second = first.model_copy(update={"id": "b", "base_price": Decimal(200)})
    snapshot = SearchSnapshot(
        search_id="feedback-test",
        status=SearchStatus.COMPLETED,
        outbound=DirectionResults(nonstop_options=[first, second]),
    )
    request = TripSearchRequest(
        origins=["STN"], destinations=["CAG"], outbound_date="2026-12-18", adults=1
    )
    SearchSessionStore(engine=engine).create(snapshot, request)
    app = FastAPI()
    app.include_router(feedback_router(engine))
    body = {
        "event_id": str(uuid4()),
        "participant_id": str(uuid4()),
        "consent": True,
        "search_id": snapshot.search_id,
        "chosen_id": "a",
        "rejected_id": "b",
        "model_version": "4",
        "profile_version": 1,
    }
    yield TestClient(app), engine, body
    engine.dispose()


def test_feedback_verified_scrubbed_and_idempotent(stored_pair):
    client, engine, body = stored_pair
    response = client.post("/api/learning/feedback", json=body)
    assert response.status_code == 201
    assert response.json()["duplicate"] is False
    assert client.post("/api/learning/feedback", json=body).json()["duplicate"] is True
    with Session(engine) as session:
        row = session.scalar(select(PreferenceFeedbackEntry))
        assert row.participant_hash != body["participant_id"]
        assert "STN" not in row.payload_json and "2026-12" not in row.payload_json
        assert json.loads(row.payload_json)["origin"] == "human"
    assert (
        client.post(
            "/api/learning/feedback", json={**body, "chosen_id": "b"}
        ).status_code
        == 409
    )


def test_feedback_rejects_unverified_options_or_no_consent(stored_pair):
    client, _, body = stored_pair
    for replacement in (
        {"chosen_id": "invented"},
        {"chosen_id": "b"},
        {"consent": False},
        {"base_price": "1"},
    ):
        assert (
            client.post(
                "/api/learning/feedback", json={**body, **replacement}
            ).status_code
            == 422
        )
    assert (
        client.post(
            "/api/learning/feedback", json={**body, "search_id": "missing"}
        ).status_code
        == 404
    )


def test_export_is_private_and_scrubbed(stored_pair, monkeypatch):
    client, _, body = stored_pair
    assert client.get("/api/learning/export").status_code == 401
    token = "test-secret-" * 5
    monkeypatch.setenv("FLIGHTLY_FEEDBACK_SYNC_TOKEN", token)
    assert (
        client.get(
            "/api/learning/export", headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    client.post("/api/learning/feedback", json=body)
    response = client.get(
        "/api/learning/export", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert len(response.json()["records"]) == 1
    assert "STN" not in response.text and body["participant_id"] not in response.text
