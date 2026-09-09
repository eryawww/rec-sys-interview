"""HTTP interface and composition root.

The algorithm is resolved from RECSYS_ALGO exactly once, at startup. The
recommendation routes perform no network I/O and need no API key.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from recsys.algorithms import build  # noqa: F401  - import registers algorithms
from recsys.algorithms.base import Rec, Recommender
from recsys.algorithms.signals import Signals
from recsys.data import Dataset, load_dataset
from recsys.explain import MissingApiKey, explain

load_dotenv()
logger = logging.getLogger("recsys")

WEB_DIR = Path(__file__).parent / "web"


@dataclass
class AppState:
    ds: Dataset
    algo: Recommender
    signals: Signals


_state: AppState | None = None


def get_state() -> AppState:
    """Return app state, building it on first use.

    The lifespan below initialises this eagerly so a bad RECSYS_ALGO
    fails at boot rather than on the first request. The lazy path covers
    callers that never run the lifespan, such as a bare TestClient.
    """
    global _state
    if _state is None:
        _state = _startup()
    return _state


class ItemOut(BaseModel):
    item_id: str
    title: str
    content_type: str
    genre: str
    score: float
    reason: str


class PopularOut(BaseModel):
    k: int
    algorithm: str
    items: list[ItemOut]


class RecommendationsOut(BaseModel):
    user_id: str
    k: int
    algorithm: str
    items: list[ItemOut]
    fallback_used: bool


class UserOut(BaseModel):
    user_id: str
    name: str
    region: str


class ExplainIn(BaseModel):
    user_id: str
    item_ids: list[str]


class ExplanationOut(BaseModel):
    item_id: str
    reason: str
    source: str


class ExplainOut(BaseModel):
    explanations: list[ExplanationOut]
    model: str
    degraded: bool


def _to_out(recs: list[Rec]) -> list[ItemOut]:
    return [
        ItemOut(
            item_id=rec.item_id,
            title=rec.title,
            content_type=rec.content_type,
            genre=rec.genre,
            score=rec.score,
            reason=rec.reason,
        )
        for rec in recs
    ]


def _startup() -> AppState:
    data_dir = Path(os.getenv("RECSYS_DATA_DIR", "data"))
    ds = load_dataset(data_dir)
    logger.info(ds.report.summary())

    name = os.getenv("RECSYS_ALGO", "item_knn")
    algo = build(name, ds)
    logger.info("Serving algorithm %r", algo.name)
    return AppState(ds=ds, algo=algo, signals=Signals(ds))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    _state = _startup()
    yield
    _state = None


app = FastAPI(title="Recommender System", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/users", response_model=list[UserOut])
def users() -> list[UserOut]:
    state = get_state()
    return [
        UserOut(user_id=u.user_id, name=u.name, region=u.region)
        for u in (state.ds.users[uid] for uid in state.ds.user_ids)
    ]


@app.get("/popular", response_model=PopularOut)
def popular(k: int = Query(10, ge=1, le=50)) -> PopularOut:
    state = get_state()
    return PopularOut(
        k=k, algorithm=state.algo.name, items=_to_out(state.algo.recommend_popular(k))
    )


@app.get("/recommendations", response_model=RecommendationsOut)
def recommendations(
    user_id: str = Query(..., min_length=1),
    k: int = Query(10, ge=1, le=50),
) -> RecommendationsOut:
    state = get_state()
    known = user_id in state.ds.users
    recs = state.algo.recommend_for_user(user_id, k)
    return RecommendationsOut(
        user_id=user_id,
        k=k,
        algorithm=state.algo.name,
        items=_to_out(recs),
        fallback_used=not known,
    )


@app.post("/explain", response_model=ExplainOut)
def explain_recommendations(body: ExplainIn) -> ExplainOut:
    state = get_state()
    if not body.item_ids:
        raise HTTPException(status_code=422, detail="item_ids must not be empty")

    # Re-derive the recommendations so the evidence is the algorithm's own,
    # never something the caller supplied.
    ranked = state.algo.recommend_for_user(body.user_id, k=50)
    wanted = set(body.item_ids)
    recs = [rec for rec in ranked if rec.item_id in wanted]
    if not recs:
        raise HTTPException(status_code=404, detail="No recommendations match those item_ids")

    user = state.ds.users.get(body.user_id)
    try:
        explanations, model, degraded = explain(
            user_name=user.name if user else "a viewer",
            genres=state.signals.top_genres(body.user_id),
            recent_titles=state.signals.recent_titles(body.user_id),
            recs=recs,
        )
    except MissingApiKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ExplainOut(
        explanations=[
            ExplanationOut(item_id=e.item_id, reason=e.reason, source=e.source)
            for e in explanations
        ],
        model=model,
        degraded=degraded,
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
