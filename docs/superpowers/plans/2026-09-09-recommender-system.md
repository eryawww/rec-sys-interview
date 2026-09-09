# Recommender System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single-command FastAPI web app serving global and personalized recommendations from three CSVs, behind a two-function algorithm interface that can be swapped by changing one environment variable.

**Architecture:** `data.py` loads CSVs into a `Dataset` and contains no recommendation logic. `algorithms/` holds every modeling decision: a `Recommender` ABC with exactly the two functions `Task.pdf` specifies, a `REGISTRY`, shared feature engineering in `signals.py`, and two implementations. `api.py` is the composition root — it resolves the algorithm from the environment once at startup and serves six routes. `explain.py` calls DeepSeek to rephrase evidence the recommender already produced, and can never affect a recommendation.

**Tech Stack:** Python 3.12+, FastAPI, uvicorn, numpy, httpx, pydantic, python-dotenv, pytest. Standard-library `csv` for loading. No pandas, no scikit-learn at runtime.

**Spec:** `docs/superpowers/specs/2026-09-09-recommender-system-design.md`

## Global Constraints

- **Interface is frozen.** `recommend_popular(k: int = 10) -> list[Rec]` and `recommend_for_user(user_id: str, k: int = 10) -> list[Rec]`. No parameters added, no wrapper layer, no service object.
- **`data.py` contains no recommendation logic.** If a change to it would need the word "recommendation" to explain, it belongs in `algorithms/`.
- **No accuracy claims anywhere** — not in code comments, docstrings, API responses, the frontend, or the README. Offline evaluation shows no algorithm beats random on this dataset (n=3,886, every 95% CI straddles zero). The default is justified by catalogue coverage (95% vs 5%) and native explainability only.
- **The signal uses `event_type` only.** `watch_seconds` is uniform noise within play/like/complete (KS p = 0.39/0.11/0.005; r = −0.009 vs interaction count). Its single legitimate use is the heavily-watched threshold.
- **Event weights:** `{complete: 3.0, like: 2.5, save: 2.0, play: 1.0, pause: 0.0, skip: -1.0}`
- **The LLM never selects, reorders or scores items.** It only rephrases a `reason` the algorithm produced.
- **Every `Rec` carries a non-empty `reason`.** LLM failure degrades to that native reason, never to `null`.
- **Secrets:** `DEEPSEEK_API_KEY` from the environment only. Never hardcoded, logged, or returned in a response.
- **Unknown `user_id` returns HTTP 200** with `fallback_used: true`, never 404.
- **`k` is validated to 1–50.**
- **Tests run offline** with no API key.

**Environment variables:** `RECSYS_ALGO` (default `item_knn`), `RECSYS_DATA_DIR` (default `data`), `RECSYS_HEAVY_WATCH_SECONDS` (default `1800`), `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` (default `deepseek-v4-flash`), `DEEPSEEK_TIMEOUT` (default `10`).

**DeepSeek contract (verified against api-docs.deepseek.com):** base URL `https://api.deepseek.com`, OpenAI-compatible `POST /chat/completions`, `response_format={"type": "json_object"}`. JSON mode **requires the word "json" in the prompt and a worked example of the output shape**. The API may return empty content — treat as failure and degrade.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Deps, pytest and ruff config |
| `.env.example` | Documents every variable, no values |
| `README.md` | One command to run; states the no-accuracy-claim finding |
| `src/recsys/__init__.py` | Empty |
| `src/recsys/data.py` | CSV → `Dataset`. Pure I/O. No recommendation logic. |
| `src/recsys/algorithms/__init__.py` | Imports implementations so they self-register |
| `src/recsys/algorithms/base.py` | `Rec`, `Recommender` ABC, `REGISTRY`, `register`, `build` |
| `src/recsys/algorithms/signals.py` | Event weights, affinity matrix, popularity vector, heavily-watched |
| `src/recsys/algorithms/popularity.py` | `PopularityRecommender` |
| `src/recsys/algorithms/item_knn.py` | `ItemKnnRecommender` (default) |
| `src/recsys/explain.py` | DeepSeek call, degradation to native reason |
| `src/recsys/api.py` | Six routes, startup wiring, static mount |
| `src/recsys/web/index.html` | Single-file frontend |
| `tests/*` | One test module per source module |

---

### Task 1: Project scaffold and data loading

**Files:**
- Create: `pyproject.toml`, `.env.example`, `src/recsys/__init__.py`, `src/recsys/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `User`, `Item`, `Event`, `LoadReport`, `Dataset`, `load_dataset(data_dir: Path) -> Dataset`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "recsys"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "numpy>=2.0",
    "httpx>=0.27",
    "pydantic>=2.8",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/recsys"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Create `.env.example`**

```bash
# Algorithm selection: popularity | item_knn
RECSYS_ALGO=item_knn
RECSYS_DATA_DIR=data
RECSYS_HEAVY_WATCH_SECONDS=1800

# Required only by POST /explain. Never commit the real value.
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_TIMEOUT=10
```

- [ ] **Step 3: Install and create the package directory**

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
mkdir -p src/recsys/algorithms src/recsys/web tests
touch src/recsys/__init__.py
```

- [ ] **Step 4: Write the failing tests**

Create `tests/test_data.py`:

```python
from pathlib import Path

import pytest

from recsys.data import Dataset, load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return load_dataset(DATA_DIR)


def test_loads_expected_row_counts(ds):
    assert len(ds.users) == 220
    assert len(ds.items) == 260
    assert len(ds.events) == 5000


def test_referential_integrity(ds):
    for event in ds.events:
        assert event.user_id in ds.users
        assert event.item_id in ds.items


def test_types_are_coerced(ds):
    event = ds.events[0]
    assert isinstance(event.watch_seconds, int)
    assert event.timestamp.year == 2025
    assert isinstance(next(iter(ds.users.values())).age, int)


def test_ids_are_sorted_and_stable(ds):
    assert ds.user_ids == sorted(ds.user_ids)
    assert ds.item_ids == sorted(ds.item_ids)


def test_missing_directory_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_dataset(tmp_path)
    assert "users.csv" in str(exc.value)


def test_malformed_rows_are_dropped_and_counted(tmp_path):
    (tmp_path / "users.csv").write_text(
        "user_id,name,age,gender,region\n"
        "u1,Ada,30,F,Jakarta\n"
        "u2,Bo,notanumber,M,Bandung\n"
    )
    (tmp_path / "items.csv").write_text(
        "item_id,title,content_type,genre\ni1,Show,series,drama\n"
    )
    (tmp_path / "events.csv").write_text(
        "user_id,item_id,event_type,watch_seconds,timestamp\n"
        "u1,i1,play,100,2025-01-01T10:00:00\n"
        "u1,i9,play,100,2025-01-01T10:00:00\n"
        "u9,i1,play,100,2025-01-01T10:00:00\n"
    )
    ds = load_dataset(tmp_path)
    assert len(ds.users) == 1
    assert len(ds.events) == 1
    assert ds.report.dropped["users_bad_field"] == 1
    assert ds.report.dropped["events_unknown_item"] == 1
    assert ds.report.dropped["events_unknown_user"] == 1
    assert "dropped" in ds.report.summary()
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'recsys.data'`

- [ ] **Step 6: Implement `src/recsys/data.py`**

```python
"""Load the three CSV files into typed records.

This module knows about files and types. It knows nothing about
recommendation: no scoring, no weighting, no similarity. Modeling
decisions live in `recsys.algorithms`.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REQUIRED_FILES = ("users.csv", "items.csv", "events.csv")


@dataclass(frozen=True, slots=True)
class User:
    user_id: str
    name: str
    age: int
    gender: str
    region: str


@dataclass(frozen=True, slots=True)
class Item:
    item_id: str
    title: str
    content_type: str
    genre: str


@dataclass(frozen=True, slots=True)
class Event:
    user_id: str
    item_id: str
    event_type: str
    watch_seconds: int
    timestamp: datetime


@dataclass
class LoadReport:
    users_read: int = 0
    items_read: int = 0
    events_read: int = 0
    dropped: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        kept = f"{self.users_read} users, {self.items_read} items, {self.events_read} events"
        if not self.dropped:
            return f"Loaded {kept}; no rows dropped."
        detail = ", ".join(f"{reason}={count}" for reason, count in sorted(self.dropped.items()))
        return f"Loaded {kept}; dropped {sum(self.dropped.values())} rows ({detail})."


@dataclass
class Dataset:
    users: dict[str, User]
    items: dict[str, Item]
    events: list[Event]
    report: LoadReport

    @property
    def user_ids(self) -> list[str]:
        return sorted(self.users)

    @property
    def item_ids(self) -> list[str]:
        return sorted(self.items)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {path.name} at {path}. "
            f"The data directory must contain: {', '.join(REQUIRED_FILES)}."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_dataset(data_dir: Path | str) -> Dataset:
    """Read users.csv, items.csv and events.csv into a Dataset.

    Rows that cannot be coerced, or that reference an unknown user or
    item, are dropped and counted in the returned LoadReport rather than
    raising. A missing file is fatal.
    """
    data_dir = Path(data_dir)
    report = LoadReport()

    users: dict[str, User] = {}
    for row in _read_rows(data_dir / "users.csv"):
        try:
            users[row["user_id"]] = User(
                user_id=row["user_id"],
                name=row["name"],
                age=int(row["age"]),
                gender=row["gender"],
                region=row["region"],
            )
        except (KeyError, TypeError, ValueError):
            report.dropped["users_bad_field"] += 1

    items: dict[str, Item] = {}
    for row in _read_rows(data_dir / "items.csv"):
        try:
            items[row["item_id"]] = Item(
                item_id=row["item_id"],
                title=row["title"],
                content_type=row["content_type"],
                genre=row["genre"],
            )
        except (KeyError, TypeError):
            report.dropped["items_bad_field"] += 1

    events: list[Event] = []
    for row in _read_rows(data_dir / "events.csv"):
        try:
            event = Event(
                user_id=row["user_id"],
                item_id=row["item_id"],
                event_type=row["event_type"],
                watch_seconds=int(row["watch_seconds"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
        except (KeyError, TypeError, ValueError):
            report.dropped["events_bad_field"] += 1
            continue
        if event.user_id not in users:
            report.dropped["events_unknown_user"] += 1
            continue
        if event.item_id not in items:
            report.dropped["events_unknown_item"] += 1
            continue
        events.append(event)

    report.users_read = len(users)
    report.items_read = len(items)
    report.events_read = len(events)
    return Dataset(users=users, items=items, events=events, report=report)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: 6 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .env.example src/recsys/__init__.py src/recsys/data.py tests/test_data.py
git commit -m "feat: load CSVs into a typed Dataset with row-level quarantine"
```

---

### Task 2: Interaction signals

**Files:**
- Create: `src/recsys/algorithms/__init__.py`, `src/recsys/algorithms/signals.py`
- Test: `tests/test_signals.py`

**Interfaces:**
- Consumes: `Dataset`, `Event` from `recsys.data`.
- Produces: `DEFAULT_EVENT_WEIGHTS: dict[str, float]`, `Signals` with attributes `item_ids: list[str]`, `item_index: dict[str, int]`, `user_index: dict[str, int]`, `affinity: np.ndarray` (users × items), `popularity: np.ndarray` (items,), and methods `heavy_mask(user_id) -> np.ndarray`, `heavily_watched(user_id, item_id) -> bool`, `watched_items(user_id) -> list[str]`, `top_genres(user_id, n=3) -> list[str]`, `recent_titles(user_id, n=3) -> list[str]`.

- [ ] **Step 1: Create the package init**

Create `src/recsys/algorithms/__init__.py` — empty for now; Task 4 fills it in.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_signals.py`:

```python
from datetime import datetime

import numpy as np
import pytest

from recsys.algorithms.signals import DEFAULT_EVENT_WEIGHTS, Signals
from recsys.data import Dataset, Event, Item, LoadReport, User


def make_dataset(events: list[tuple[str, str, str, int]]) -> Dataset:
    users = {u: User(u, f"Name {u}", 30, "F", "Jakarta") for u in ("u1", "u2")}
    items = {i: Item(i, f"Title {i}", "series", "drama") for i in ("i1", "i2", "i3")}
    parsed = [
        Event(u, i, kind, secs, datetime(2025, 1, 1 + n, 12, 0))
        for n, (u, i, kind, secs) in enumerate(events)
    ]
    return Dataset(users=users, items=items, events=parsed, report=LoadReport())


def test_weights_come_from_event_type_only():
    ds = make_dataset([("u1", "i1", "complete", 10), ("u1", "i2", "play", 3000)])
    sig = Signals(ds)
    assert sig.affinity[sig.user_index["u1"], sig.item_index["i1"]] == 3.0
    assert sig.affinity[sig.user_index["u1"], sig.item_index["i2"]] == 1.0


def test_skip_lowers_score():
    ds = make_dataset([("u1", "i1", "play", 100), ("u1", "i1", "skip", 5)])
    sig = Signals(ds)
    assert sig.affinity[sig.user_index["u1"], sig.item_index["i1"]] == 0.0


def test_popularity_is_column_sum():
    ds = make_dataset([("u1", "i1", "like", 10), ("u2", "i1", "save", 10)])
    sig = Signals(ds)
    assert sig.popularity[sig.item_index["i1"]] == pytest.approx(4.5)


def test_heavily_watched_on_complete_regardless_of_seconds():
    ds = make_dataset([("u1", "i1", "complete", 1)])
    sig = Signals(ds)
    assert sig.heavily_watched("u1", "i1") is True


def test_heavily_watched_on_accumulated_seconds():
    ds = make_dataset([("u1", "i1", "play", 1000), ("u1", "i1", "play", 900)])
    sig = Signals(ds, heavy_watch_seconds=1800)
    assert sig.heavily_watched("u1", "i1") is True


def test_skipped_item_stays_recommendable():
    ds = make_dataset([("u1", "i1", "skip", 20)])
    sig = Signals(ds)
    assert sig.heavily_watched("u1", "i1") is False


def test_heavy_mask_matches_scalar_check():
    ds = make_dataset([("u1", "i1", "complete", 10), ("u1", "i2", "skip", 5)])
    sig = Signals(ds)
    mask = sig.heavy_mask("u1")
    assert mask[sig.item_index["i1"]]
    assert not mask[sig.item_index["i2"]]
    assert mask.dtype == np.bool_


def test_unknown_user_has_empty_mask_and_history():
    ds = make_dataset([("u1", "i1", "play", 10)])
    sig = Signals(ds)
    assert not sig.heavy_mask("nobody").any()
    assert sig.watched_items("nobody") == []
    assert sig.recent_titles("nobody") == []


def test_recent_titles_are_newest_first():
    ds = make_dataset(
        [("u1", "i1", "play", 10), ("u1", "i2", "play", 10), ("u1", "i3", "play", 10)]
    )
    sig = Signals(ds)
    assert sig.recent_titles("u1", n=2) == ["Title i3", "Title i2"]


def test_default_weights_are_event_type_only():
    assert DEFAULT_EVENT_WEIGHTS == {
        "complete": 3.0,
        "like": 2.5,
        "save": 2.0,
        "play": 1.0,
        "pause": 0.0,
        "skip": -1.0,
    }
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_signals.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'recsys.algorithms.signals'`

- [ ] **Step 4: Implement `src/recsys/algorithms/signals.py`**

```python
"""Feature engineering shared by the shipped algorithms.

These are modeling choices, not facts about the file format, which is why
they live under `algorithms/` rather than in `data.py`. An algorithm is
free to ignore this module and read `Dataset.events` directly.

The signal is a function of `event_type` alone. `watch_seconds` is not
used as a strength signal: within play/like/complete it is
indistinguishable from Uniform(120, 3600) (KS p = 0.39/0.11/0.005) and
correlates -0.009 with an item's interaction count, so weighting by it
would weight noise. Its one use here is the heavily-watched threshold,
which is a behavioural definition rather than a measure of interest.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict

import numpy as np

from recsys.data import Dataset

DEFAULT_EVENT_WEIGHTS: dict[str, float] = {
    "complete": 3.0,
    "like": 2.5,
    "save": 2.0,
    "play": 1.0,
    "pause": 0.0,
    "skip": -1.0,
}

DEFAULT_HEAVY_WATCH_SECONDS = int(os.getenv("RECSYS_HEAVY_WATCH_SECONDS", "1800"))


class Signals:
    """Affinity and popularity arrays derived from a Dataset."""

    def __init__(
        self,
        ds: Dataset,
        weights: dict[str, float] | None = None,
        heavy_watch_seconds: int | None = None,
    ) -> None:
        self.ds = ds
        self.weights = dict(weights or DEFAULT_EVENT_WEIGHTS)
        self.heavy_watch_seconds = (
            DEFAULT_HEAVY_WATCH_SECONDS if heavy_watch_seconds is None else heavy_watch_seconds
        )

        self.user_ids = ds.user_ids
        self.item_ids = ds.item_ids
        self.user_index = {u: n for n, u in enumerate(self.user_ids)}
        self.item_index = {i: n for n, i in enumerate(self.item_ids)}

        n_users, n_items = len(self.user_ids), len(self.item_ids)
        self.affinity = np.zeros((n_users, n_items), dtype=np.float64)

        seconds: defaultdict[tuple[str, str], int] = defaultdict(int)
        completed: set[tuple[str, str]] = set()
        history: defaultdict[str, list[str]] = defaultdict(list)

        for event in ds.events:
            row = self.user_index[event.user_id]
            col = self.item_index[event.item_id]
            self.affinity[row, col] += self.weights.get(event.event_type, 0.0)

            pair = (event.user_id, event.item_id)
            seconds[pair] += event.watch_seconds
            if event.event_type == "complete":
                completed.add(pair)
            history[event.user_id].append(event.item_id)

        self.popularity = self.affinity.sum(axis=0)
        self._seconds = dict(seconds)
        self._completed = completed
        # Events arrive in file order; the loader preserves it, and the
        # fixture is chronological, so the tail is the most recent.
        self._history = {user: list(dict.fromkeys(reversed(seen))) for user, seen in history.items()}

        self._heavy_masks: dict[str, np.ndarray] = {}

    def heavily_watched(self, user_id: str, item_id: str) -> bool:
        pair = (user_id, item_id)
        if pair in self._completed:
            return True
        return self._seconds.get(pair, 0) >= self.heavy_watch_seconds

    def heavy_mask(self, user_id: str) -> np.ndarray:
        """Boolean array over item_ids: True where the item is heavily watched."""
        cached = self._heavy_masks.get(user_id)
        if cached is not None:
            return cached
        mask = np.zeros(len(self.item_ids), dtype=bool)
        for item_id in self._history.get(user_id, ()):
            if self.heavily_watched(user_id, item_id):
                mask[self.item_index[item_id]] = True
        self._heavy_masks[user_id] = mask
        return mask

    def watched_items(self, user_id: str) -> list[str]:
        """Distinct item ids the user interacted with, most recent first."""
        return list(self._history.get(user_id, ()))

    def recent_titles(self, user_id: str, n: int = 3) -> list[str]:
        return [self.ds.items[i].title for i in self.watched_items(user_id)[:n]]

    def top_genres(self, user_id: str, n: int = 3) -> list[str]:
        counts = Counter(self.ds.items[i].genre for i in self.watched_items(user_id))
        return [genre for genre, _ in counts.most_common(n)]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_signals.py -v`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add src/recsys/algorithms/__init__.py src/recsys/algorithms/signals.py tests/test_signals.py
git commit -m "feat: event-type interaction signal and heavily-watched definition"
```

---

### Task 3: The Recommender seam

**Files:**
- Create: `src/recsys/algorithms/base.py`
- Test: `tests/test_base.py`

**Interfaces:**
- Consumes: `Dataset`, `Item` from `recsys.data`.
- Produces: `Rec` (frozen dataclass: `item_id: str`, `title: str`, `content_type: str`, `genre: str`, `score: float`, `reason: str`), `Recommender` ABC with `name: ClassVar[str]`, `__init__(ds)`, abstract `recommend_popular(k=10)`, abstract `recommend_for_user(user_id, k=10)`, protected helper `_rec(item_id, score, reason) -> Rec`; module globals `REGISTRY: dict[str, type[Recommender]]`, `register(cls)`, `build(name, ds) -> Recommender`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_base.py`:

```python
import pytest

from recsys.algorithms.base import REGISTRY, Rec, Recommender, build, register
from recsys.data import Dataset, Item, LoadReport, User


@pytest.fixture
def ds() -> Dataset:
    return Dataset(
        users={"u1": User("u1", "Ada", 30, "F", "Jakarta")},
        items={"i1": Item("i1", "Show One", "series", "drama")},
        events=[],
        report=LoadReport(),
    )


def test_rec_is_immutable():
    rec = Rec("i1", "Show One", "series", "drama", 1.0, "because")
    with pytest.raises(AttributeError):
        rec.score = 2.0  # type: ignore[misc]


def test_rec_helper_fills_item_fields(ds):
    class Stub(Recommender):
        name = "stub_helper"

        def recommend_popular(self, k=10):
            return [self._rec("i1", 1.0, "because")]

        def recommend_for_user(self, user_id, k=10):
            return self.recommend_popular(k)

    rec = Stub(ds).recommend_popular()[0]
    assert rec == Rec("i1", "Show One", "series", "drama", 1.0, "because")


def test_registry_round_trip(ds):
    @register
    class Registered(Recommender):
        name = "stub_registered"

        def recommend_popular(self, k=10):
            return []

        def recommend_for_user(self, user_id, k=10):
            return []

    assert REGISTRY["stub_registered"] is Registered
    assert isinstance(build("stub_registered", ds), Registered)
    del REGISTRY["stub_registered"]


def test_build_unknown_name_lists_valid_options(ds):
    with pytest.raises(ValueError) as exc:
        build("does_not_exist", ds)
    message = str(exc.value)
    assert "does_not_exist" in message
    assert "item_knn" in message


def test_abstract_methods_are_enforced(ds):
    class Incomplete(Recommender):
        name = "stub_incomplete"

    with pytest.raises(TypeError):
        Incomplete(ds)  # type: ignore[abstract]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'recsys.algorithms.base'`

- [ ] **Step 3: Implement `src/recsys/algorithms/base.py`**

```python
"""The swappable seam.

The interface is exactly the two functions Task.pdf 2.1.2 and 2.1.3
specify. Nothing wraps them and nothing decomposes them: an algorithm is
one file implementing both, registered by name and selected with the
RECSYS_ALGO environment variable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from recsys.data import Dataset


@dataclass(frozen=True, slots=True)
class Rec:
    """One recommended item.

    `reason` is the algorithm's own explanation, computed while ranking.
    It is always populated, which is what lets the LLM layer be a pure
    upgrade rather than a dependency.
    """

    item_id: str
    title: str
    content_type: str
    genre: str
    score: float
    reason: str


class Recommender(ABC):
    name: ClassVar[str]

    def __init__(self, ds: Dataset) -> None:
        self.ds = ds

    @abstractmethod
    def recommend_popular(self, k: int = 10) -> list[Rec]:
        """Return the top-k items by global popularity, highest first.

        If k is larger than the catalogue, return every item.
        """

    @abstractmethod
    def recommend_for_user(self, user_id: str, k: int = 10) -> list[Rec]:
        """Return a ranked list of recommendations for the given user.

        Does not recommend items the user has already heavily watched.
        If the user is unknown or has no history, falls back to
        recommend_popular(k).
        """

    def _rec(self, item_id: str, score: float, reason: str) -> Rec:
        item = self.ds.items[item_id]
        return Rec(
            item_id=item.item_id,
            title=item.title,
            content_type=item.content_type,
            genre=item.genre,
            score=float(score),
            reason=reason,
        )


REGISTRY: dict[str, type[Recommender]] = {}


def register(cls: type[Recommender]) -> type[Recommender]:
    REGISTRY[cls.name] = cls
    return cls


def build(name: str, ds: Dataset) -> Recommender:
    try:
        cls = REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "<none registered>"
        raise ValueError(
            f"Unknown algorithm {name!r}. Registered algorithms: {known}. "
            f"Set RECSYS_ALGO to one of these."
        ) from None
    return cls(ds)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_base.py -v`
Expected: `test_build_unknown_name_lists_valid_options` FAILS (no `item_knn` registered yet); the other 4 pass. This test is completed by Task 5.

- [ ] **Step 5: Commit**

```bash
git add src/recsys/algorithms/base.py tests/test_base.py
git commit -m "feat: Recommender ABC and algorithm registry"
```

---

### Task 4: Popularity algorithm and the shared contract suite

**Files:**
- Create: `src/recsys/algorithms/popularity.py`
- Modify: `src/recsys/algorithms/__init__.py`
- Test: `tests/test_algorithms.py`

**Interfaces:**
- Consumes: `Recommender`, `Rec`, `register` from `base`; `Signals` from `signals`.
- Produces: `PopularityRecommender` with `name = "popularity"`. `tests/test_algorithms.py` defines the contract suite parametrized over `REGISTRY.values()` — every later algorithm inherits it with no edit to the test file.

- [ ] **Step 1: Write the failing contract tests**

Create `tests/test_algorithms.py`:

```python
"""Contract every registered algorithm must satisfy.

Parametrized over REGISTRY, so adding an algorithm adds its coverage
automatically. Do not add algorithm-specific assertions here.
"""

from pathlib import Path

import pytest

import recsys.algorithms  # noqa: F401  - triggers registration
from recsys.algorithms.base import REGISTRY, Rec
from recsys.algorithms.signals import Signals
from recsys.data import load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def ds():
    return load_dataset(DATA_DIR)


@pytest.fixture(scope="module")
def signals(ds):
    return Signals(ds)


@pytest.fixture(params=sorted(REGISTRY), scope="module")
def algo(request, ds):
    return REGISTRY[request.param](ds)


def test_registry_is_not_empty():
    assert REGISTRY


def test_popular_returns_k_items(algo):
    assert len(algo.recommend_popular(k=5)) == 5


def test_popular_k_larger_than_catalogue_returns_all(algo, ds):
    assert len(algo.recommend_popular(k=10_000)) == len(ds.items)


def test_popular_sorted_descending(algo):
    scores = [r.score for r in algo.recommend_popular(k=20)]
    assert scores == sorted(scores, reverse=True)


def test_popular_has_no_duplicates(algo):
    recs = algo.recommend_popular(k=50)
    assert len({r.item_id for r in recs}) == len(recs)


def test_popular_is_identical_for_every_caller(algo):
    assert algo.recommend_popular(k=10) == algo.recommend_popular(k=10)


def test_for_user_respects_k(algo, ds):
    user_id = ds.user_ids[0]
    assert len(algo.recommend_for_user(user_id, k=5)) == 5


def test_for_user_sorted_descending(algo, ds):
    scores = [r.score for r in algo.recommend_for_user(ds.user_ids[0], k=20)]
    assert scores == sorted(scores, reverse=True)


def test_for_user_has_no_duplicates(algo, ds):
    recs = algo.recommend_for_user(ds.user_ids[0], k=50)
    assert len({r.item_id for r in recs}) == len(recs)


def test_for_user_excludes_heavily_watched(algo, ds, signals):
    for user_id in ds.user_ids[:25]:
        for rec in algo.recommend_for_user(user_id, k=20):
            assert not signals.heavily_watched(user_id, rec.item_id)


def test_unknown_user_falls_back_to_popular(algo):
    assert algo.recommend_for_user("u999", k=5) == algo.recommend_popular(k=5)


def test_every_rec_carries_a_reason(algo, ds):
    for rec in algo.recommend_for_user(ds.user_ids[0], k=10):
        assert isinstance(rec, Rec)
        assert rec.reason.strip()
    for rec in algo.recommend_popular(k=10):
        assert rec.reason.strip()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_algorithms.py -v`
Expected: FAIL at collection or `test_registry_is_not_empty` — nothing is registered yet.

- [ ] **Step 3: Implement `src/recsys/algorithms/popularity.py`**

```python
"""Global popularity: the same ranking for everyone.

Also the fallback every other algorithm uses for an unknown user, and the
baseline the personalized algorithms are read against.
"""

from __future__ import annotations

import numpy as np

from recsys.algorithms.base import Rec, Recommender, register
from recsys.algorithms.signals import Signals
from recsys.data import Dataset


@register
class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self, ds: Dataset) -> None:
        super().__init__(ds)
        self.signals = Signals(ds)
        # Stable ordering: break score ties by item_id so the ranking is
        # reproducible across runs.
        order = np.lexsort((np.array(self.signals.item_ids), -self.signals.popularity))
        self._order = order
        self._rank_of = {self.signals.item_ids[idx]: n + 1 for n, idx in enumerate(order)}

    def _reason(self, item_id: str) -> str:
        return f"Popular right now - #{self._rank_of[item_id]} across all viewers"

    def recommend_popular(self, k: int = 10) -> list[Rec]:
        chosen = self._order[:k]
        return [
            self._rec(self.signals.item_ids[idx], self.signals.popularity[idx],
                      self._reason(self.signals.item_ids[idx]))
            for idx in chosen
        ]

    def recommend_for_user(self, user_id: str, k: int = 10) -> list[Rec]:
        if user_id not in self.ds.users:
            return self.recommend_popular(k)
        mask = self.signals.heavy_mask(user_id)
        recs: list[Rec] = []
        for idx in self._order:
            if mask[idx]:
                continue
            item_id = self.signals.item_ids[idx]
            recs.append(self._rec(item_id, self.signals.popularity[idx], self._reason(item_id)))
            if len(recs) == k:
                break
        return recs
```

- [ ] **Step 4: Register it on import**

Replace `src/recsys/algorithms/__init__.py` with:

```python
"""Importing this package registers every shipped algorithm."""

from recsys.algorithms import item_knn, popularity  # noqa: F401
from recsys.algorithms.base import REGISTRY, Rec, Recommender, build, register

__all__ = ["REGISTRY", "Rec", "Recommender", "build", "register"]
```

Note: `item_knn` does not exist until Task 5. Until then, temporarily import only `popularity`; Task 5 restores the line above.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_algorithms.py -v`
Expected: 12 passed (one parametrization, `popularity`)

- [ ] **Step 6: Commit**

```bash
git add src/recsys/algorithms/popularity.py src/recsys/algorithms/__init__.py tests/test_algorithms.py
git commit -m "feat: popularity algorithm and shared contract test suite"
```

---

### Task 5: Item-kNN algorithm

**Files:**
- Create: `src/recsys/algorithms/item_knn.py`
- Modify: `src/recsys/algorithms/__init__.py`
- Test: `tests/test_item_knn.py`

**Interfaces:**
- Consumes: `Recommender`, `Rec`, `register`, `Signals`, `PopularityRecommender`.
- Produces: `ItemKnnRecommender` with `name = "item_knn"` and `TOP_NEIGHBOURS = 50`. Automatically joins the Task 4 contract suite.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_item_knn.py`:

```python
from pathlib import Path

import pytest

from recsys.algorithms.item_knn import ItemKnnRecommender
from recsys.algorithms.signals import Signals
from recsys.data import load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def ds():
    return load_dataset(DATA_DIR)


@pytest.fixture(scope="module")
def algo(ds):
    return ItemKnnRecommender(ds)


def test_output_differs_between_users(ds, algo):
    a = [r.item_id for r in algo.recommend_for_user(ds.user_ids[0], k=10)]
    b = [r.item_id for r in algo.recommend_for_user(ds.user_ids[1], k=10)]
    assert a != b


def test_reason_names_a_watched_title(ds, algo):
    signals = Signals(ds)
    user_id = ds.user_ids[0]
    watched = {ds.items[i].title for i in signals.watched_items(user_id)}
    reasons = [r.reason for r in algo.recommend_for_user(user_id, k=5)]
    assert any(any(title in reason for title in watched) for reason in reasons)


def test_covers_far_more_of_the_catalogue_than_popularity(ds, algo):
    from recsys.algorithms.popularity import PopularityRecommender

    pop = PopularityRecommender(ds)
    knn_items = {r.item_id for u in ds.user_ids for r in algo.recommend_for_user(u, k=10)}
    pop_items = {r.item_id for u in ds.user_ids for r in pop.recommend_for_user(u, k=10)}
    assert len(knn_items) > 5 * len(pop_items)


def test_similarity_matrix_has_zero_diagonal(algo):
    import numpy as np

    assert np.allclose(np.diag(algo.similarity), 0.0)


def test_user_with_no_usable_neighbours_still_gets_k_items(ds, algo):
    recs = algo.recommend_for_user(ds.user_ids[0], k=260)
    assert len(recs) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_item_knn.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'recsys.algorithms.item_knn'`

- [ ] **Step 3: Implement `src/recsys/algorithms/item_knn.py`**

```python
"""Item-based k-nearest-neighbours over the interaction matrix.

The score for an item is a sum of contributions from items the user
already engaged with, which means the largest contributing term names the
item responsible for the recommendation. That per-item provenance is the
native `reason`, and it is the evidence the LLM layer rephrases.

Neighbours are truncated to the top TOP_NEIGHBOURS per item: the
truncated form both ranked higher and covered more of the catalogue than
the untruncated form in offline evaluation.
"""

from __future__ import annotations

import numpy as np

from recsys.algorithms.base import Rec, Recommender, register
from recsys.algorithms.popularity import PopularityRecommender
from recsys.algorithms.signals import Signals
from recsys.data import Dataset


@register
class ItemKnnRecommender(Recommender):
    name = "item_knn"
    TOP_NEIGHBOURS = 50

    def __init__(self, ds: Dataset) -> None:
        super().__init__(ds)
        self.signals = Signals(ds)
        self._fallback = PopularityRecommender(ds)

        matrix = self.signals.affinity
        norms = np.linalg.norm(matrix, axis=0)
        norms[norms == 0.0] = 1.0
        normalised = matrix / norms
        similarity = normalised.T @ normalised
        np.fill_diagonal(similarity, 0.0)

        # Keep only the strongest TOP_NEIGHBOURS per item.
        n_items = similarity.shape[0]
        keep = min(self.TOP_NEIGHBOURS, n_items - 1)
        threshold = np.partition(similarity, n_items - keep, axis=1)[:, n_items - keep][:, None]
        similarity[similarity < threshold] = 0.0
        self.similarity = similarity

    def recommend_popular(self, k: int = 10) -> list[Rec]:
        return self._fallback.recommend_popular(k)

    def recommend_for_user(self, user_id: str, k: int = 10) -> list[Rec]:
        if user_id not in self.signals.user_index:
            return self.recommend_popular(k)

        row = self.signals.affinity[self.signals.user_index[user_id]]
        if not row.any():
            return self.recommend_popular(k)

        # contributions[j, i] is how much watched item j pushed up item i.
        contributions = row[:, None] * self.similarity
        scores = contributions.sum(axis=0)

        scores[self.signals.heavy_mask(user_id)] = -np.inf
        order = np.argsort(-scores)

        recs: list[Rec] = []
        for idx in order:
            if len(recs) == k:
                break
            score = scores[idx]
            if not np.isfinite(score) or score <= 0.0:
                break
            source = int(np.argmax(contributions[:, idx]))
            source_title = self.ds.items[self.signals.item_ids[source]].title
            recs.append(
                self._rec(self.signals.item_ids[idx], score, f"Because you watched {source_title}")
            )

        if len(recs) < k:
            recs.extend(self._top_up(user_id, k - len(recs), {r.item_id for r in recs}))
        return recs

    def _top_up(self, user_id: str, needed: int, already: set[str]) -> list[Rec]:
        """Fill the tail with popular items when similarity runs out.

        A user whose neighbours are all heavily watched can exhaust the
        positive-score candidates before reaching k. Returning fewer than
        k would violate the interface, so popularity completes the list.
        """
        mask = self.signals.heavy_mask(user_id)
        filler: list[Rec] = []
        for rec in self._fallback.recommend_popular(len(self.signals.item_ids)):
            if len(filler) == needed:
                break
            idx = self.signals.item_index[rec.item_id]
            if mask[idx] or rec.item_id in already:
                continue
            filler.append(rec)
        return filler
```

- [ ] **Step 4: Restore the full package init**

`src/recsys/algorithms/__init__.py`:

```python
"""Importing this package registers every shipped algorithm."""

from recsys.algorithms import item_knn, popularity  # noqa: F401
from recsys.algorithms.base import REGISTRY, Rec, Recommender, build, register

__all__ = ["REGISTRY", "Rec", "Recommender", "build", "register"]
```

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: all pass, including `tests/test_base.py::test_build_unknown_name_lists_valid_options` (now that `item_knn` is registered) and the contract suite running twice — once per algorithm.

- [ ] **Step 6: Commit**

```bash
git add src/recsys/algorithms/item_knn.py src/recsys/algorithms/__init__.py tests/test_item_knn.py
git commit -m "feat: item-kNN algorithm with native per-item reasons"
```

---

### Task 6: API — health, users, popular, recommendations

**Files:**
- Create: `src/recsys/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `load_dataset`, `build`, `REGISTRY`, `Signals`.
- Produces: `app: FastAPI`, `get_state() -> AppState` where `AppState` has `.ds`, `.algo`, `.signals`. Routes `GET /health`, `GET /users`, `GET /popular`, `GET /recommendations`, and a static mount at `/`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api.py`:

```python
from fastapi.testclient import TestClient

from recsys.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_users_listing_shape():
    body = client.get("/users").json()
    assert len(body) == 220
    assert set(body[0]) == {"user_id", "name", "region"}


def test_popular_returns_k_items():
    body = client.get("/popular?k=5").json()
    assert body["k"] == 5
    assert len(body["items"]) == 5
    assert body["algorithm"] in {"popularity", "item_knn"}
    assert set(body["items"][0]) == {
        "item_id", "title", "content_type", "genre", "score", "reason"
    }


def test_popular_is_the_same_for_every_request():
    first = client.get("/popular?k=10").json()["items"]
    second = client.get("/popular?k=10").json()["items"]
    assert first == second


def test_recommendations_for_known_user():
    body = client.get("/recommendations?user_id=u1&k=5").json()
    assert body["user_id"] == "u1"
    assert body["fallback_used"] is False
    assert len(body["items"]) == 5
    assert all(item["reason"] for item in body["items"])


def test_unknown_user_returns_200_with_fallback_flag():
    response = client.get("/recommendations?user_id=u999&k=5")
    assert response.status_code == 200
    body = response.json()
    assert body["fallback_used"] is True
    assert body["items"] == client.get("/popular?k=5").json()["items"]


def test_k_out_of_range_is_rejected():
    assert client.get("/popular?k=0").status_code == 422
    assert client.get("/popular?k=51").status_code == 422
    assert client.get("/recommendations?user_id=u1&k=0").status_code == 422


def test_missing_user_id_is_rejected():
    assert client.get("/recommendations").status_code == 422


def test_root_serves_the_frontend():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'recsys.api'`

- [ ] **Step 3: Create a placeholder frontend so the static mount resolves**

```bash
mkdir -p src/recsys/web
printf '<!doctype html><title>Recommender</title><h1>placeholder</h1>' > src/recsys/web/index.html
```

- [ ] **Step 4: Implement `src/recsys/api.py`**

```python
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

from recsys.algorithms import REGISTRY, build  # noqa: F401  - import registers algorithms
from recsys.algorithms.base import Rec, Recommender
from recsys.algorithms.signals import Signals
from recsys.data import Dataset, load_dataset

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
    if _state is None:
        raise RuntimeError("Application state is not initialised.")
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


def _to_out(recs: list[Rec]) -> list[ItemOut]:
    return [ItemOut(**vars(rec)) for rec in recs]


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
    return PopularOut(k=k, algorithm=state.algo.name, items=_to_out(state.algo.recommend_popular(k)))


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


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: 9 passed

- [ ] **Step 6: Verify the server actually starts**

Run: `uv run uvicorn recsys.api:app --port 8000` in one shell, then:
`curl -s localhost:8000/recommendations?user_id=u1\&k=3`
Expected: JSON with three items, each with a non-empty `reason`. Stop the server.

- [ ] **Step 7: Commit**

```bash
git add src/recsys/api.py src/recsys/web/index.html tests/test_api.py
git commit -m "feat: HTTP API for popular and personalized recommendations"
```

---

### Task 7: LLM explanation layer

**Files:**
- Create: `src/recsys/explain.py`
- Modify: `src/recsys/api.py` (add the `/explain` route)
- Test: `tests/test_explain.py`

**Interfaces:**
- Consumes: `Rec`, `Signals`, `Dataset`.
- Produces: `Explanation` (frozen dataclass: `item_id: str`, `reason: str`, `source: str` — `"llm"` or `"native"`), `MissingApiKey(Exception)`, `build_messages(user_name, genres, recent_titles, recs) -> list[dict]`, `explain(user_name, genres, recent_titles, recs, *, client=None) -> tuple[list[Explanation], str, bool]` returning `(explanations, model, degraded)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_explain.py`:

```python
import json

import httpx
import pytest

from recsys.algorithms.base import Rec
from recsys.explain import Explanation, MissingApiKey, build_messages, explain

RECS = [
    Rec("i1", "Start-Up", "series", "drama", 2.0, "Because you watched Crash Landing on You"),
    Rec("i2", "Vincenzo", "series", "thriller", 1.5, "Because you watched Start-Up"),
]


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.deepseek.com")


def reply(payload: dict) -> httpx.Response:
    body = {"choices": [{"message": {"content": json.dumps(payload)}}]}
    return httpx.Response(200, json=body)


def test_prompt_mentions_json_and_shows_an_example():
    messages = build_messages("Ada", ["drama"], ["Crash Landing on You"], RECS)
    blob = " ".join(m["content"] for m in messages)
    assert "json" in blob.lower()
    assert "explanations" in blob
    assert "Start-Up" in blob


def test_prompt_carries_the_native_reason_as_evidence():
    messages = build_messages("Ada", ["drama"], [], RECS)
    blob = " ".join(m["content"] for m in messages)
    assert "Because you watched Crash Landing on You" in blob


def test_successful_call_returns_llm_reasons():
    def handler(request):
        return reply({"explanations": [
            {"item_id": "i1", "reason": "You keep coming back to drama series."},
            {"item_id": "i2", "reason": "A thriller with the same lead."},
        ]})

    out, model, degraded = explain("Ada", ["drama"], [], RECS, client=make_client(handler))
    assert degraded is False
    assert [e.source for e in out] == ["llm", "llm"]
    assert out[0].reason == "You keep coming back to drama series."
    assert model


def test_timeout_degrades_to_native_reasons():
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    out, _, degraded = explain("Ada", ["drama"], [], RECS, client=make_client(handler))
    assert degraded is True
    assert [e.reason for e in out] == [r.reason for r in RECS]
    assert all(e.source == "native" for e in out)


def test_http_error_degrades():
    out, _, degraded = explain(
        "Ada", ["drama"], [], RECS,
        client=make_client(lambda request: httpx.Response(500, text="boom")),
    )
    assert degraded is True
    assert all(e.source == "native" for e in out)


def test_malformed_json_degrades():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    out, _, degraded = explain("Ada", ["drama"], [], RECS, client=make_client(handler))
    assert degraded is True
    assert all(e.source == "native" for e in out)


def test_empty_content_degrades():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    out, _, degraded = explain("Ada", ["drama"], [], RECS, client=make_client(handler))
    assert degraded is True


def test_partial_response_degrades_only_the_missing_item():
    def handler(request):
        return reply({"explanations": [{"item_id": "i1", "reason": "Drama, your usual."}]})

    out, _, degraded = explain("Ada", ["drama"], [], RECS, client=make_client(handler))
    assert degraded is True
    assert out[0].source == "llm"
    assert out[1].source == "native"
    assert out[1].reason == RECS[1].reason


def test_llm_cannot_introduce_or_reorder_items():
    def handler(request):
        return reply({"explanations": [
            {"item_id": "i2", "reason": "second"},
            {"item_id": "i1", "reason": "first"},
            {"item_id": "i999", "reason": "not recommended at all"},
        ]})

    out, _, _ = explain("Ada", ["drama"], [], RECS, client=make_client(handler))
    assert [e.item_id for e in out] == ["i1", "i2"]


def test_overlong_reason_is_truncated():
    def handler(request):
        return reply({"explanations": [{"item_id": "i1", "reason": "x" * 500}]})

    out, _, _ = explain("Ada", ["drama"], [], RECS[:1], client=make_client(handler))
    assert len(out[0].reason) <= 160


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(MissingApiKey):
        explain("Ada", ["drama"], [], RECS)


def test_explanation_is_immutable():
    e = Explanation("i1", "why", "llm")
    with pytest.raises(AttributeError):
        e.reason = "other"  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_explain.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'recsys.explain'`

- [ ] **Step 3: Implement `src/recsys/explain.py`**

```python
"""Rephrase, with an LLM, evidence the recommender already produced.

The model is never asked to choose, score or order items. It receives the
ranked list and each item's native reason and returns wording. Anything
it says about an item that was not recommended is discarded, and the
order of the input list is preserved regardless of the order it replies
in. This keeps the recommender falsifiable: the two explanation layers
cannot disagree about what was recommended or why.

Every failure mode degrades to the native reason, so a caller always
receives one explanation per recommendation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import httpx

from recsys.algorithms.base import Rec

logger = logging.getLogger("recsys.explain")

BASE_URL = "https://api.deepseek.com"
MAX_REASON_CHARS = 160

EXAMPLE = '{"explanations": [{"item_id": "i1", "reason": "You often watch drama, so this may fit."}]}'


class MissingApiKey(RuntimeError):
    """DEEPSEEK_API_KEY is not set."""


@dataclass(frozen=True, slots=True)
class Explanation:
    item_id: str
    reason: str
    source: str  # "llm" | "native"


def build_messages(
    user_name: str,
    genres: list[str],
    recent_titles: list[str],
    recs: list[Rec],
) -> list[dict[str, str]]:
    """Assemble the chat messages.

    DeepSeek's JSON mode requires the word "json" and a worked example of
    the output shape in the prompt, both of which are present below.
    """
    catalogue = "\n".join(
        f"- {rec.item_id} | {rec.title} | {rec.genre} | evidence: {rec.reason}" for rec in recs
    )
    context = [f"Viewer: {user_name}"]
    if genres:
        context.append(f"Watches mostly: {', '.join(genres)}")
    if recent_titles:
        context.append(f"Recently watched: {', '.join(recent_titles)}")

    system = (
        "You write one-sentence explanations for streaming recommendations. "
        "You are given items that have already been chosen and the evidence "
        "for each. Rephrase that evidence warmly and specifically. "
        "Never add, remove, reorder or rank items. "
        "Reply with json only, in exactly this shape: " + EXAMPLE
    )
    user = (
        "\n".join(context)
        + "\n\nRecommended items:\n"
        + catalogue
        + "\n\nReturn json with one entry per item_id above, each reason "
        f"under {MAX_REASON_CHARS} characters."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _native(recs: list[Rec]) -> list[Explanation]:
    return [Explanation(rec.item_id, rec.reason, "native") for rec in recs]


def explain(
    user_name: str,
    genres: list[str],
    recent_titles: list[str],
    recs: list[Rec],
    *,
    client: httpx.Client | None = None,
) -> tuple[list[Explanation], str, bool]:
    """Return (explanations, model, degraded), one entry per rec, in order."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise MissingApiKey(
            "DEEPSEEK_API_KEY is not set. Recommendations still work; "
            "only generated explanations are unavailable."
        )
    if not recs:
        return [], os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"), False

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "10"))
    owns_client = client is None
    client = client or httpx.Client(base_url=BASE_URL, timeout=timeout)

    payload = {
        "model": model,
        "messages": build_messages(user_name, genres, recent_titles, recs),
        "response_format": {"type": "json_object"},
        "max_tokens": 1000,
        "temperature": 0.7,
    }

    try:
        response = client.post(
            "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise ValueError("empty content")
        parsed = json.loads(content)
        by_id = {
            entry["item_id"]: str(entry["reason"])
            for entry in parsed["explanations"]
            if entry.get("item_id") and entry.get("reason")
        }
    except Exception as exc:  # noqa: BLE001 - every failure degrades identically
        logger.warning("Explanation call failed (%s); using native reasons.", type(exc).__name__)
        return _native(recs), model, True
    finally:
        if owns_client:
            client.close()

    out: list[Explanation] = []
    degraded = False
    for rec in recs:
        text = by_id.get(rec.item_id)
        if text:
            out.append(Explanation(rec.item_id, text[:MAX_REASON_CHARS], "llm"))
        else:
            out.append(Explanation(rec.item_id, rec.reason, "native"))
            degraded = True
    return out, model, degraded
```

- [ ] **Step 4: Add the `/explain` route to `src/recsys/api.py`**

Add these imports near the existing ones:

```python
from recsys.explain import MissingApiKey, explain
```

Add these models beside the other response models:

```python
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
```

Add this route after `recommendations`:

```python
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
        explanations=[ExplanationOut(**vars(e)) for e in explanations],
        model=model,
        degraded=degraded,
    )
```

- [ ] **Step 5: Add API-level tests for the route**

Append to `tests/test_api.py`:

```python
def test_explain_without_key_returns_503(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    items = client.get("/recommendations?user_id=u1&k=3").json()["items"]
    response = client.post(
        "/explain", json={"user_id": "u1", "item_ids": [i["item_id"] for i in items]}
    )
    assert response.status_code == 503
    assert "DEEPSEEK_API_KEY" in response.json()["detail"]


def test_explain_rejects_empty_item_ids():
    assert client.post("/explain", json={"user_id": "u1", "item_ids": []}).status_code == 422


def test_explain_rejects_items_that_were_not_recommended():
    response = client.post("/explain", json={"user_id": "u1", "item_ids": ["nope"]})
    assert response.status_code == 404
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_explain.py tests/test_api.py -v`
Expected: 12 + 12 passed

- [ ] **Step 7: Commit**

```bash
git add src/recsys/explain.py src/recsys/api.py tests/test_explain.py tests/test_api.py
git commit -m "feat: DeepSeek explanation layer degrading to native reasons"
```

---

### Task 8: Frontend

**Files:**
- Modify: `src/recsys/web/index.html` (replaces the placeholder)
- Test: manual, plus the existing `test_root_serves_the_frontend`

**Interfaces:**
- Consumes: `GET /popular`, `GET /users`, `GET /recommendations`, `POST /explain`.
- Produces: nothing other modules import.

- [ ] **Step 1: Write `src/recsys/web/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recommender</title>
<style>
  :root {
    --bg: #0f1115; --card: #181b22; --line: #262a33;
    --fg: #e8eaed; --muted: #9aa0aa; --accent: #7aa2f7;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 1000px; margin: 0 auto; padding: 32px 20px 80px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 28px; }
  h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .08em;
       color: var(--muted); margin: 36px 0 14px; }
  .grid { display: grid; gap: 12px;
          grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); }
  .card { background: var(--card); border: 1px solid var(--line);
          border-radius: 10px; padding: 14px; }
  .title { font-weight: 600; margin-bottom: 6px; }
  .meta { color: var(--muted); font-size: 12px; text-transform: capitalize; }
  .reason { margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--line);
            font-size: 13px; color: var(--muted); }
  .reason.llm { color: var(--fg); }
  .picker { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  input, select {
    background: var(--card); color: var(--fg); border: 1px solid var(--line);
    border-radius: 8px; padding: 9px 12px; font: inherit; min-width: 260px;
  }
  select { min-width: 90px; }
  .banner { background: #2b2415; border: 1px solid #4a3d1d; color: #e8d9a8;
            border-radius: 8px; padding: 10px 12px; font-size: 13px; margin-bottom: 14px; }
  .note { color: var(--muted); font-size: 12px; margin-top: 8px; }
</style>
</head>
<body>
<main>
  <h1>Recommender</h1>
  <div class="sub" id="algo">&nbsp;</div>

  <h2>Popular now</h2>
  <div class="grid" id="popular"></div>

  <h2>Recommended for a viewer</h2>
  <div class="picker">
    <input id="user" list="users" placeholder="Type a name or user id, e.g. u1 or Sari"
           autocomplete="off">
    <datalist id="users"></datalist>
    <select id="k">
      <option>5</option><option selected>10</option><option>20</option>
    </select>
  </div>
  <div class="note" id="status"></div>
  <div id="personal"></div>
</main>

<script>
const $ = (id) => document.getElementById(id);
let users = [];

const card = (item) => {
  const el = document.createElement("div");
  el.className = "card";
  el.dataset.itemId = item.item_id;
  el.innerHTML =
    `<div class="title"></div>
     <div class="meta"></div>
     <div class="reason"></div>`;
  el.querySelector(".title").textContent = item.title;
  el.querySelector(".meta").textContent = `${item.content_type} · ${item.genre}`;
  el.querySelector(".reason").textContent = item.reason;
  return el;
};

const grid = (items) => {
  const g = document.createElement("div");
  g.className = "grid";
  items.forEach((item) => g.appendChild(card(item)));
  return g;
};

async function loadPopular() {
  const res = await fetch("/popular?k=10");
  const body = await res.json();
  $("algo").textContent = `Serving algorithm: ${body.algorithm}`;
  const target = $("popular");
  target.replaceChildren(...[...grid(body.items).children]);
}

async function loadUsers() {
  users = await (await fetch("/users")).json();
  const list = $("users");
  users.forEach((u) => {
    const opt = document.createElement("option");
    opt.value = u.user_id;
    opt.label = `${u.name} — ${u.region}`;
    list.appendChild(opt);
  });
}

function resolveUserId(raw) {
  const text = raw.trim();
  if (!text) return null;
  const byId = users.find((u) => u.user_id.toLowerCase() === text.toLowerCase());
  if (byId) return byId.user_id;
  const byName = users.find((u) => u.name.toLowerCase() === text.toLowerCase());
  return byName ? byName.user_id : text;
}

async function loadPersonal() {
  const userId = resolveUserId($("user").value);
  const panel = $("personal");
  if (!userId) { panel.replaceChildren(); $("status").textContent = ""; return; }

  $("status").textContent = "Loading…";
  const k = $("k").value;
  const body = await (await fetch(
    `/recommendations?user_id=${encodeURIComponent(userId)}&k=${k}`)).json();

  panel.replaceChildren();
  if (body.fallback_used) {
    const warn = document.createElement("div");
    warn.className = "banner";
    warn.textContent =
      `No history for "${userId}" — showing globally popular titles instead.`;
    panel.appendChild(warn);
  }
  const g = grid(body.items);
  panel.appendChild(g);
  $("status").textContent = "";

  // Recommendations are already explained. The LLM call upgrades the
  // wording in place and is allowed to fail silently.
  try {
    const res = await fetch("/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId, item_ids: body.items.map((i) => i.item_id),
      }),
    });
    if (!res.ok) return;
    const out = await res.json();
    out.explanations.forEach((e) => {
      const node = g.querySelector(`[data-item-id="${CSS.escape(e.item_id)}"] .reason`);
      if (node && e.source === "llm") {
        node.textContent = e.reason;
        node.classList.add("llm");
      }
    });
  } catch (_) { /* native reasons remain on screen */ }
}

let timer;
$("user").addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(loadPersonal, 250);
});
$("k").addEventListener("change", loadPersonal);

loadPopular();
loadUsers();
</script>
</body>
</html>
```

- [ ] **Step 2: Verify manually**

Run: `uv run uvicorn recsys.api:app --port 8000` and open `http://localhost:8000`.

Confirm, in order:
1. "Popular now" renders 10 cards immediately, each with a reason.
2. Typing `u1` (or `Sari`) shows suggestions and then a "Recommended for a viewer" grid appears *below* the popular grid, which stays on screen.
3. Every personalized card has a reason before any LLM call completes.
4. With a valid `DEEPSEEK_API_KEY`, reasons brighten as the LLM wording replaces the native text.
5. Typing `u999` shows the amber fallback banner and popular titles.

- [ ] **Step 3: Run the suite**

Run: `uv run pytest -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/recsys/web/index.html
git commit -m "feat: single-file frontend, global first then personalized"
```

---

### Task 9: README and end-to-end verification

**Files:**
- Create: `README.md`
- Test: full suite plus a clean-start check

**Interfaces:**
- Consumes: everything.
- Produces: nothing.

- [ ] **Step 1: Write `README.md`**

````markdown
# Recommendation System

A web app over a streaming catalogue: global popular recommendations,
personalized recommendations per user, and a one-line reason for every
item.

## Run it

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run uvicorn recsys.api:app --port 8000
```

Open <http://localhost:8000>.

Generated explanations additionally need a key:

```bash
cp .env.example .env   # then set DEEPSEEK_API_KEY
```

Without one, everything still works — each recommendation falls back to
the reason the algorithm itself produced.

## API

| Method | Path | Example |
|---|---|---|
| GET | `/health` | `curl localhost:8000/health` |
| GET | `/users` | `curl localhost:8000/users` |
| GET | `/popular?k=` | `curl 'localhost:8000/popular?k=5'` |
| GET | `/recommendations?user_id=&k=` | `curl 'localhost:8000/recommendations?user_id=u1&k=5'` |
| POST | `/explain` | `curl -X POST localhost:8000/explain -H 'Content-Type: application/json' -d '{"user_id":"u1","item_ids":["i1"]}'` |

An unknown `user_id` returns 200 with `"fallback_used": true` and the
globally popular list.

## Swapping the algorithm

```bash
RECSYS_ALGO=popularity uv run uvicorn recsys.api:app --port 8000
```

Shipped: `popularity`, `item_knn` (default).

To add one, create `src/recsys/algorithms/<name>.py`:

```python
from recsys.algorithms.base import Recommender, register

@register
class MyRecommender(Recommender):
    name = "mine"

    def recommend_popular(self, k=10): ...
    def recommend_for_user(self, user_id, k=10): ...
```

Import it in `src/recsys/algorithms/__init__.py` and set
`RECSYS_ALGO=mine`. The contract test suite in `tests/test_algorithms.py`
picks it up automatically — no test file changes.

## How watch history becomes a signal

The interaction score is a function of `event_type` alone:

```
complete 3.0 · like 2.5 · save 2.0 · play 1.0 · pause 0.0 · skip -1.0
```

`watch_seconds` is deliberately excluded. In this dataset it is a
function of the event type rather than of engagement — within
play/like/complete it is indistinguishable from Uniform(120, 3600)
(KS p = 0.39/0.11/0.005) and correlates −0.009 with an item's interaction
count. Weighting by it would weight noise. It is used only to decide
whether an item counts as "heavily watched" and should be withheld
(`RECSYS_HEAVY_WATCH_SECONDS`, default 1800, or any `complete` event).

## On accuracy

Nine algorithms were evaluated offline (5-fold CV, n = 3,886 held-out
interactions, HR@10 and NDCG@10, paired bootstrap against a random
baseline). **None separates from random** — every 95% confidence interval
straddles zero. Corroborating diagnostics: per-user genre concentration
matches shuffled data (HHI 0.1847 vs 0.1848), demographic association
χ²/dof is 0.98–1.20, and a user's own history predicts their next genre
*worse* than the global mode (0.180 vs 0.244).

This is a property of the provided `events.csv`, whose interactions are
consistent with uniform random sampling of (user, item) pairs.

`item_knn` is the default for two reasons that are measurable here, and
accuracy is not one of them:

- **Coverage** — it surfaces ~95% of the catalogue and gives different
  users different lists, where popularity surfaces ~5% and gives everyone
  the same list.
- **Explainability** — its score is a sum over the user's watched items,
  so the largest term names the title responsible: *"Because you watched
  Start-Up."*

The analysis is in `docs/recommender-research.md`; the scripts that
produce it are in `research/`.

## Explanations

Two layers over one evidence object:

1. **The algorithm's own reason**, computed while ranking and returned by
   `/recommendations`. Always present, no network required.
2. **The LLM's phrasing of that same evidence**, from `POST /explain`.

The model is given the chosen items and their evidence and asked only for
wording. It never selects, scores or reorders — anything it returns for
an item that was not recommended is discarded. If the call fails, times
out, returns malformed JSON, or omits an item, that item keeps its native
reason and the response reports `"degraded": true`.

## Tests

```bash
uv run pytest
```

Runs offline; no API key needed.
````

- [ ] **Step 2: Verify a clean start from scratch**

```bash
rm -rf .venv
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest
uv run uvicorn recsys.api:app --port 8000 &
sleep 3
curl -sf localhost:8000/health
curl -sf 'localhost:8000/popular?k=3'
curl -sf 'localhost:8000/recommendations?user_id=u1&k=3'
curl -sf 'localhost:8000/recommendations?user_id=u999&k=3'
kill %1
```

Expected: tests pass; `/health` returns `{"status":"ok"}`; both
recommendation calls return three items each with a non-empty `reason`;
the `u999` call reports `"fallback_used": true`.

- [ ] **Step 3: Verify the algorithm swap works**

```bash
RECSYS_ALGO=popularity uv run uvicorn recsys.api:app --port 8001 &
sleep 3
curl -s 'localhost:8001/recommendations?user_id=u1&k=3' | grep -o '"algorithm":"[a-z_]*"'
kill %1
```

Expected: `"algorithm":"popularity"`.

- [ ] **Step 4: Verify an invalid algorithm fails clearly**

```bash
RECSYS_ALGO=nonsense uv run uvicorn recsys.api:app --port 8002
```

Expected: startup fails with `Unknown algorithm 'nonsense'. Registered algorithms: item_knn, popularity.`

- [ ] **Step 5: Lint**

```bash
uv run ruff check src tests
uv run ruff format --check src tests
```

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: README with run instructions, signal design and evaluation findings"
```

---

## Self-Review

**Spec coverage.** §2 interface → Task 3. §3 boundaries → Tasks 1–5 file
placement. §4.1/4.2 signal → Task 2. §5 evaluation claims → Task 9
README. §6 algorithms → Tasks 4–5. §7 API → Tasks 6–7. §8 two-layer
explanation → Task 7. §9 frontend → Task 8. §10 configuration → Task 1
`.env.example`. §11 failure behaviour → Tasks 1 (CSV, malformed), 3
(unknown algorithm), 6 (unknown user, `k` validation), 7 (LLM failures).
§12 testing → every task. No gaps.

**Known ordering wrinkle, handled explicitly.** Task 3 Step 4 leaves one
test red (`test_build_unknown_name_lists_valid_options` asserts `item_knn`
appears in the error message) and Task 4 Step 4 imports a module that does
not yet exist. Both are called out in their steps and are closed by Task
5 Step 5. This is deliberate: the alternative is asserting on a weaker
error message that would not survive the real registry.

**Type consistency.** `Rec(item_id, title, content_type, genre, score,
reason)` is constructed only through `Recommender._rec` and consumed by
`_to_out` in `api.py` and `build_messages` in `explain.py` — field names
match at all three sites. `Signals.heavy_mask`, `.watched_items`,
`.recent_titles`, `.top_genres`, `.item_index`, `.item_ids`, `.affinity`,
`.popularity` are defined in Task 2 and used under those exact names in
Tasks 4, 5 and 7. `explain()` returns `(list[Explanation], str, bool)` and
is unpacked as three values in `api.py`.
