# Recommender System — Design Spec

Status: approved for implementation
Date: 2026-09-09
Sources: `data/Task.pdf` (core), `data/Task_2.pdf` (LLM integration),
`docs/recommender-research.md` (dataset analysis and algorithm survey)

## 1. Goal

A web app over a streaming catalogue producing two deliverables:

1. **Global recommendations** — identical for every visitor.
2. **Personalized recommendations** — dependent on a chosen user.

Each personalized recommendation carries a short reason.

The organising constraint is that the recommendation algorithm is
**swappable**: replacing it means adding one file and changing one
environment variable, with no edit to the data layer, the API, or the
frontend.

## 2. Interface — the whole seam

The abstraction is exactly the two functions `Task.pdf` §2.1.2 and §2.1.3
specify. Nothing wraps them and nothing decomposes them.

```python
# src/recsys/algorithms/base.py
class Recommender(ABC):
    name: str

    def __init__(self, ds: Dataset) -> None:
        self.ds = ds          # precompute here; constructed once at startup

    @abstractmethod
    def recommend_popular(self, k: int = 10) -> list[Rec]:
        """Top-k items by global popularity, highest first.
        Each Rec includes at least item_id and title.
        If k exceeds the catalogue, return every item."""

    @abstractmethod
    def recommend_for_user(self, user_id: str, k: int = 10) -> list[Rec]:
        """Ranked recommendations for user_id, highest first.
        Excludes items the user has already heavily watched.
        Unknown user or no history -> return recommend_popular(k)."""

REGISTRY: dict[str, type[Recommender]] = {}
```

`Rec` is a frozen dataclass: `item_id`, `title`, `content_type`, `genre`,
`score`, `reason` — where `reason` is the algorithm's own explanation of
why this item was ranked (§8), not LLM text.

**Adding an algorithm.** Create `algorithms/<name>.py`, subclass
`Recommender`, register it, set `RECSYS_ALGO=<name>`. All precomputation
belongs in `__init__`.

**Contract, enforced by a shared parametrized test suite that every
registered algorithm inherits automatically:**

1. Results are sorted by `score` descending.
2. No duplicate `item_id` in a result.
3. `len(result) <= k`, and equals `min(k, catalogue_size)` for
   `recommend_popular`.
4. `recommend_for_user` never raises on an unknown id — it returns the
   popular list.
5. `recommend_for_user` never returns an item the user has heavily watched.
6. Every `Rec` carries a non-empty `reason`.

## 3. Module boundaries

```
src/recsys/
├─ data.py                # CSV -> Dataset. Pure I/O.
├─ algorithms/
│  ├─ base.py             # Recommender ABC + REGISTRY
│  ├─ signals.py          # event weights, affinity matrix, heavily_watched
│  ├─ popularity.py
│  └─ item_knn.py
├─ explain.py             # DeepSeek call
├─ api.py                 # FastAPI routes + static mount
└─ web/index.html         # single-file frontend
```

Dependency direction is one-way: `data → algorithms → api`, with `explain`
used only by `api`. `algorithms/` never imports `api` or `explain`.

**`data.py` contains no recommendation logic.** Its whole job: read three
CSVs, coerce types, drop and count invalid rows, expose lookups by id. The
test of the boundary is that its responsibility can be stated without the
word "recommendation".

**`algorithms/signals.py` holds the modeling choices** — event weights,
the affinity matrix, the heavily-watched threshold. These are hypotheses
about what watch history means, not facts about the file format, so they
sit on the algorithm side of the boundary where an implementation can
decline to use them. An algorithm ranking by raw event counts imports
`Dataset` and ignores `signals` entirely.

## 4. Data

`users.csv` 220 rows (`user_id, name, age, gender, region`), `items.csv`
260 rows (`item_id, title, content_type, genre`), `events.csv` 5,000 rows
(`user_id, item_id, event_type, watch_seconds, timestamp`).

Referential integrity holds: every id in `events.csv` resolves, every user
has 12–39 events, every item has ≥10 events. No cold users and no cold
items exist in the file, so the fallback path is reachable only through an
id absent from the data.

Matrix density is 8.4% (4,783 distinct user–item pairs of 57,200). Repeat
interactions are 4.3% of events. Timestamps span 2025-01-01 to 2025-03-01.
There is no missing data in any column.

### 4.1 `watch_seconds` carries no interest information

`watch_seconds` is a function of `event_type`, not of engagement:

| event_type | n | median `watch_seconds` |
|---|---:|---:|
| play | 2,529 | 1,866 |
| complete | 770 | 1,863 |
| like | 474 | 1,829 |
| pause | 509 | 181 |
| save | 248 | 100 |
| skip | 470 | 22 |

Within play / like / complete the value is indistinguishable from
Uniform(≈120, 3600) — KS test against uniform gives p = 0.39 (like), 0.11
(complete), 0.005 (play). An item's mean `watch_seconds` correlates
**−0.009** with its interaction count. `items.csv` has no duration column,
so seconds cannot be converted to a completion ratio either.

The brief's stated assumption — *"`watch_seconds` = implicit feedback
strength"* — does not hold on this data. **Any weighting built on raw
seconds is weighting noise.** The information lives entirely in
`event_type`.

This is why the signal below has no seconds term. `watch_seconds` is used
in exactly one place — the heavily-watched threshold — where it functions
as a behavioural definition the brief asks the implementer to choose,
not as a strength signal.

### 4.2 Signal definition (`algorithms/signals.py`)

```
W = {complete: 3.0, like: 2.5, save: 2.0, play: 1.0, pause: 0.0, skip: -1.0}

affinity[u][i]  = Σ W[e.event_type]  over that user's events on that item
popularity[i]   = Σ W[e.event_type]  over all events on that item

heavily_watched(u, i) = any `complete` event on (u, i)
                        OR Σ watch_seconds on (u, i) >= HEAVY_WATCH_SECONDS
```

`W` and `HEAVY_WATCH_SECONDS` (default 1800) are configuration. Items
merely skipped or paused stay recommendable, matching the brief's wording
of "heavily watched" rather than "seen".

Recency decay and Bayesian shrinkage are documented improvements to the
popularity score (`docs/recommender-research.md` §3.1) deliberately left
out of the first build to keep the scoring one line to explain. They enter
as new registered algorithms, not as edits to this one.

## 5. Offline evaluation and what it licenses us to claim

Nine algorithms evaluated under 5-fold cross-validation over positive
interactions (n = 3,886 held-out events), HR@10 and NDCG@10 against a
random baseline, 4,000-sample paired bootstrap.

| model | HR@10 | NDCG@10 | vs random | 95% CI | significant |
|---|---:|---:|---:|---|---|
| svd_16 | 0.0443 | 0.0210 | +0.0028 | [−0.0062, +0.0116] | no |
| content | 0.0461 | 0.0206 | +0.0046 | [−0.0046, +0.0136] | no |
| userknn | 0.0414 | 0.0204 | 0.0000 | [−0.0082, +0.0090] | no |
| **random** | 0.0414 | 0.0193 | — | — | — |
| itemknn_top50 | 0.0399 | 0.0184 | −0.0015 | [−0.0108, +0.0072] | no |
| popularity | 0.0378 | 0.0182 | −0.0036 | [−0.0124, +0.0057] | no |
| itemknn_full | 0.0386 | 0.0178 | −0.0028 | [−0.0116, +0.0057] | no |
| hybrid | 0.0396 | 0.0176 | −0.0018 | [−0.0103, +0.0069] | no |
| svd_64 | 0.0350 | 0.0154 | −0.0064 | [−0.0147, +0.0018] | no |

**No algorithm separates from random.** Every confidence interval
straddles zero; popularity and item-kNN both score below the random
baseline. Independent diagnostics agree:

- Per-user genre concentration (HHI) = 0.1847; the same statistic on
  shuffled item assignments = 0.1848. Users are no more genre-focused
  than chance.
- Demographic association χ²/dof: age 1.08, gender 0.98, region 1.20,
  where 1.0 is no association.
- Genre transition χ²/dof = 1.24. Predicting a user's next genre from the
  mode of their own history scores **0.180**, against **0.244** for
  ignoring the user and guessing the globally most common genre.

The interactions are consistent with uniform random sampling of
(user, item) pairs. This is a property of `events.csv`, not a defect in
any implementation, and no model choice recovers a signal that is absent.

**What follows for this build.** Accuracy cannot select a default, so the
default is chosen on grounds that remain measurable and honest:

- **Catalogue coverage**, from the same run: `popularity` surfaces 5% of
  the catalogue and returns an identical list to every user;
  `itemknn_top50` surfaces 95% and returns visibly different lists per
  user. The requirement is *personalized* recommendations, and only one of
  these produces them.
- **Native explainability.** Item-kNN's score is a sum over the user's
  watched items; the largest contributing term names the item responsible.
  No other family yields a per-item reason without post-hoc
  reconstruction.

**`RECSYS_ALGO` defaults to `item_knn`.** The README states plainly that
the choice rests on coverage and explainability and that offline accuracy
on this dataset is indistinguishable from random for every candidate.
**No accuracy claim appears anywhere in the product.**

## 6. Algorithms shipped

**`popularity`** — ranks by `popularity[i]`. `recommend_for_user` returns
the same ranking minus heavily-watched items. Also the fallback for every
other algorithm, and the interpretability baseline.
Reason text: `"Popular right now — #3 across all viewers"`.

**`item_knn`** (default) — cosine similarity between item columns of the
affinity matrix, truncated to the top 50 neighbours per item (the
truncated variant beat the full one on both NDCG@10 and coverage).

```
score(u, i) = Σ over j in items(u) of  affinity[u][j] * cos(i, j)
```

The largest contributing `j` is retained per recommended item.
Reason text: `"Because you watched Start-Up"`.

`docs/recommender-research.md` §3.4 argues for a factorization algorithm
(SVD/ALS) as a third registration, on the grounds that hosting a
fit/persist/load model proves the abstraction generalises. That is a
documented extension, not part of this build — the seam is designed to
accept it without modification.

## 7. API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{"status": "ok"}` |
| GET | `/users` | `[{user_id, name, region}]` — feeds the autocomplete |
| GET | `/popular?k=10` | `{k, algorithm, items[]}` |
| GET | `/recommendations?user_id=&k=10` | `{user_id, k, algorithm, items[], fallback_used}` |
| POST | `/explain` | `{user_id, item_ids[]}` → `{explanations[], model, degraded}` |
| GET | `/` | `web/index.html` |

`k` is validated to 1–50. An unknown `user_id` returns **200 with
`fallback_used: true`**, not 404 — matching the `u999` example in
`Task.pdf` §2.2.

`/popular` and `/recommendations` perform no network I/O and require no
API key.

### 7.1 Request path

```
GET /recommendations?user_id=u1&k=5
  1. validate params      pydantic     k in [1,50]                ← 422 on bad input
  2. dispatch             api.py       REGISTRY[RECSYS_ALGO]      ← resolved once at startup
  3. rank                 item_knn     260 items scored           ← the swappable component
  4. exclude heavy-watch  item_knn     260 → 247
  5. truncate             item_knn     247 → 5
  6. cold-start guard     item_knn     unknown user → popular(k), fallback_used = true
  → { user_id, k, algorithm, items[5], fallback_used: false }
     each item already carries its native `reason`

Execution model: synchronous, in-memory, no network. Matrices are built
once at startup, so a request is arithmetic over preloaded arrays and
cannot fail on a missing API key or a slow upstream.

POST /explain  { user_id, item_ids[5] }
  1. gather evidence      api.py       native reason + user's genre mix + titles
  2. one batched call     DeepSeek     json_object, timeout 10 s, 1 retry
  3. validate             pydantic     one reason per item_id, <= 160 chars
  4. degrade per item     explain.py   on any failure → the native reason
  → { explanations[5], model, degraded }

Execution model: one round-trip covering all k items, on its own endpoint.
The frontend has already painted the recommendations before this is
issued, so LLM latency never blocks the product.
```

## 8. Explanation — two layers over one evidence object

Interpretability has two tracks over a single structure:

- **The algorithm's own reason**, computed during ranking and returned by
  `/recommendations`. Item-kNN names the neighbour item that contributed
  most; popularity names the rank. This is derived from the arithmetic,
  cannot disagree with the ranking, and needs no network.
- **The LLM's phrasing of that same evidence**, from `/explain`.

**The LLM explains a decision it did not make.** It receives the evidence
the recommender produced and is asked to phrase it. It is never given the
freedom to select, reorder, or score items. Violating this makes the
recommender unfalsifiable and lets the two explanations contradict each
other.

**Degradation is therefore already designed**: when the LLM call fails,
the native reason is the displayed text. There is no "explanation
unavailable" state and no `null` reason — `/recommendations` alone always
yields a complete, explained result.

### 8.1 `explain.py`

DeepSeek over its OpenAI-compatible surface:
`POST https://api.deepseek.com/chat/completions`,
`response_format={"type": "json_object"}`.

- Key from `DEEPSEEK_API_KEY`. Never hardcoded, never logged, never
  returned in a response.
- Model from `DEEPSEEK_MODEL`. The exact identifier is verified against
  the provider's published model list at implementation time.
- Context sent: the user's dominant genres and content types, three recent
  titles, and for each candidate its title, genre and native reason.
  Nothing else. No RAG, no vector store, no agent loop.
- One request covers all `k` items.

**Failure handling.** Missing key → `503`, and the frontend keeps showing
native reasons. Timeout, HTTP error, unparseable JSON, or an item missing
from the response → `200`, native reason substituted for the affected
items, `degraded: true`.

## 9. Frontend (`web/index.html`)

One file, vanilla JavaScript, no build step, served by FastAPI.

```
1. page load     GET /popular?k=10       → "Popular Now" grid, always visible
2. user picker   GET /users (once)       → text input filtered client-side,
                                            suggestions on every keystroke
3. user chosen   GET /recommendations    → "For You" grid appears below,
                                            native reasons rendered immediately
                 POST /explain           → reasons upgraded in place to LLM text
```

The global section renders first and is never removed — the personalized
section is appended beneath it, so both deliverables are on screen
simultaneously. A `fallback_used: true` response renders a banner
explaining that the user is unknown and popular items are shown.

Because every card arrives with a reason already, the LLM upgrade is
purely additive: no skeletons, no empty states, nothing breaks if
`/explain` never returns.

## 10. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `RECSYS_ALGO` | `item_knn` | Selects the registered algorithm |
| `RECSYS_DATA_DIR` | `data` | Location of the three CSVs |
| `RECSYS_HEAVY_WATCH_SECONDS` | `1800` | Heavily-watched threshold |
| `DEEPSEEK_API_KEY` | — | From `.env`; required only by `/explain` |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | Explanation model |
| `DEEPSEEK_TIMEOUT` | `10` | Seconds |

`.env` is gitignored. `.env.example` documents every variable with no
values.

## 11. Failure behaviour

| Condition | Behaviour |
|---|---|
| Missing or unreadable CSV at startup | Fail immediately, naming the path and what was expected. No traceback. |
| Malformed row (bad integer, unresolvable id) | Dropped, counted, summarised in one startup log line. |
| `RECSYS_ALGO` not in `REGISTRY` | Startup failure listing registered names. |
| Unknown `user_id` | 200, popular items, `fallback_used: true`. |
| `k` greater than catalogue size | Return every available item. |
| `k` outside 1–50 | 422 from validation. |
| `DEEPSEEK_API_KEY` unset | `/explain` → 503; native reasons remain on screen. |
| LLM timeout or malformed JSON | 200, native reason substituted, `degraded: true`. |

## 12. Testing

- `test_data.py` — loading, coercion, malformed-row quarantine,
  referential integrity.
- `test_signals.py` — event weighting, `skip` lowers score,
  `heavily_watched` threshold behaviour.
- `test_algorithms.py` — the six contract rules, parametrized over
  `REGISTRY.values()`, so a new algorithm inherits them without touching
  the test file. Cold start and `k > catalogue` covered here.
- `test_explain.py` — fake HTTP transport: success, timeout, malformed
  JSON, missing key, item missing from response. Asserts the native reason
  survives every failure. No network access.
- `test_api.py` — every route, fallback response shape, validation errors.

The suite runs offline and requires no API key.

## 13. Out of scope

Facet filtering by `content_type` or `genre` (`Task.pdf` §2.3, optional),
a watch-history endpoint, per-request algorithm switching, a factorization
algorithm, recency decay, Bayesian shrinkage, hybrid blending, user
accounts, persistence, and a CLI. `Task.pdf` §2.2 requires one interface
style; the Web API is that choice.
