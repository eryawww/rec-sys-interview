# Recommender algorithms available for this dataset

Research output. Scope: which recommendation approaches are *usable* given the
data in `data/`, and what each one can and cannot deliver. No implementation.

---

## 1. What the data is

| | |
|---|---|
| users | 220 (`user_id, name, age, gender, region`) |
| items | 260 (`item_id, title, content_type, genre`) |
| events | 5,000 (`user_id, item_id, event_type, watch_seconds, timestamp`) |
| period | 2025-01-01 → 2025-03-01 (60 days) |
| missing values | none, in any column |
| unknown IDs | none — every event's user and item resolve |
| cold users / cold items | 0 / 0 — every user and item appears in events |
| distinct (user, item) pairs | 4,783 → matrix density **8.4 %** |
| events per user | min 12, median 22, max 39 |
| events per item | min 10, median 19, max 31 |
| repeat interactions | 217 (4.3 %) |

`event_type` distribution: play 2529, complete 770, pause 509, like 474, skip 470, save 248.

`content_type`: series 115, movie 91, microdrama 35, tv 19.
`genre`: romance 66, thriller 38, drama 32, anime 31, family 28, action 24, comedy 22, kids 19.

**Item catalogue has no duration field.** This matters: `watch_seconds` cannot be
converted into completion ratio, which is the standard way to make watch time
comparable across a 90-minute movie and a 3-minute microdrama.

### 1.1 `watch_seconds` is a function of `event_type`, not of interest

| event_type | min | median | max |
|---|---|---|---|
| play | 122 | 1866 | 3600 |
| like | 135 | 1829 | 3589 |
| complete | 135 | 1863 | 3596 |
| pause | 60 | 181 | 300 |
| save | 30 | 100 | 180 |
| skip | 5 | 22 | 40 |

Within play / like / complete the value is indistinguishable from
Uniform(≈120, 3600) (KS test vs uniform: p = 0.39 for like, 0.11 for complete,
0.005 for play). Correlation between an item's mean `watch_seconds` and its
interaction count is **−0.009**.

Consequence: the task brief's stated assumption — *"`watch_seconds` = implicit
feedback strength (more = stronger interest)"* — does not hold here. Summing
`watch_seconds` per item is arithmetically almost the same ranking as counting
play/like/complete events, plus uniform noise. Any weighting scheme built on
raw seconds is weighting noise. The information lives in `event_type`.

### 1.2 There is no personalization signal to learn

Three independent tests, all negative:

- **Genre concentration.** Mean per-user genre HHI = 0.1847. Same statistic on
  the data with item assignments shuffled = 0.1848. Users are not more
  genre-focused than chance. Same result for `content_type` (0.3785 vs 0.3789).
- **Demographics.** χ²/dof for genre preference by attribute: age 1.08,
  gender 0.98, region 1.20. A ratio of 1.0 is "no association".
- **Sequence.** Genre-to-genre transition χ²/dof = 1.24. Predicting a user's
  next genre from the mode of their own history is **0.180 accurate — worse
  than the 0.244 you get by always guessing the globally most common genre.**
  There are also no sessions: median gap between a user's events is 41 hours,
  only 2 % of gaps are under an hour.

### 1.3 Benchmark: nothing beats random

Temporal leave-last-out (each user's final positive event held out), ranking all
unseen items, n = 208 evaluable users.

| model | HR@10 | NDCG@10 | catalogue coverage |
|---|---|---|---|
| userKNN cosine | 0.0625 | 0.0314 | 0.57 |
| itemKNN cosine, top-50 neighbours | 0.0577 | 0.0305 | 0.95 |
| popularity, recency-decayed | 0.0481 | 0.0243 | 0.05 |
| popularity, event count | 0.0529 | 0.0235 | 0.05 |
| hybrid content + popularity | 0.0433 | 0.0214 | 0.32 |
| popularity, Σ watch_seconds | 0.0433 | 0.0210 | 0.06 |
| TruncatedSVD, 64 factors | 0.0481 | 0.0208 | 1.00 |
| **random** | **0.0385** | **0.0174** | **1.00** |
| content-based (genre + content_type) | 0.0337 | 0.0166 | 0.70 |
| implicit ALS, 32 factors | 0.0288 | 0.0136 | 0.99 |
| SVD, 8–32 factors | 0.019–0.024 | 0.010–0.011 | 0.67–0.96 |

Repeated over 10 random leave-one-out splits, HR@10 (mean ± sd):

```
random       0.0456 ± 0.0088
itemKNN      0.0480 ± 0.0097
popularity   0.0446 ± 0.0152
userKNN      0.0373 ± 0.0140
```

Paired bootstrap (5,000 resamples) of each model minus random: every 95 %
confidence interval straddles zero. **No approach is measurably better than
shuffling the catalogue.**

This is a property of the data, not of the algorithms. The interactions are
consistent with uniform random sampling of (user, item) pairs — which is what
a synthetic fixture looks like.

---

## 2. What this means for algorithm selection

Offline accuracy cannot discriminate between candidates here. Any claim of the
form "model X recommends better" is unfalsifiable on this dataset. Two things
follow.

**Selection criterion is not accuracy.** It is: does the algorithm degrade
sanely, is its output explainable, does it cover the catalogue, and does it stay
correct when a real dataset with real signal is dropped in. The task brief's own
review criteria point the same way — it lists signal design, code clarity,
end-to-end operation and edge-case handling, and never mentions a metric.

**The evaluation harness is itself a deliverable.** Being able to state "these
five algorithms are indistinguishable from random on the provided fixture, here
is the bootstrap" is a stronger result than picking one and asserting it works.
It also becomes the switch-testing mechanism the pluggable architecture needs.

---

## 3. Algorithm families, assessed against this data

### 3.1 Popularity — for `recommend_popular`

The global recommender is not really a modelling choice; it is a scoring-policy
choice. Options, all computable from `events.csv` alone:

| variant | definition | property |
|---|---|---|
| interaction count | number of events per item | ignores engagement quality |
| Σ watch_seconds | total seconds per item | **on this data, noise-weighted counting — see §1.1** |
| event-weighted score | Σ w(event_type), e.g. complete 3, like 2.5, save 2, play 1, pause 0, skip −1 | uses the only column that carries information here |
| completion count | count of `complete` events only | sharpest quality signal, 770 events to work with |
| recency-decayed | weight × 0.5^(age / half-life) | surfaces trending, needs a half-life constant |
| Wilson / Bayesian shrunk | smooths items with few events toward the prior mean | prevents a 10-event item topping the chart on luck |

Recommendation: **event-type-weighted count with recency decay and Bayesian
shrinkage.** All three components are defensible without appealing to accuracy,
each is one line to explain to a user, and the weights are the plug point for
iteration. Coverage is inherently ~5 % of the catalogue, which is correct
behaviour for a global chart.

### 3.2 Item-based kNN (cosine on the interaction matrix) — for `recommend_for_user`

Score(u, i) = Σ_{j ∈ history(u)} sim(i, j) · weight(u, j), neighbours truncated
to top-N.

- Fits the data volume exactly: a 220 × 260 matrix at 8.4 % density is a
  57k-cell dense array; the full item-item similarity is 260 × 260. Recomputing
  from scratch takes milliseconds — no training step, no model artefact, no
  serving/training skew.
- **It is the only family that produces a native explanation.** The top
  contributing `j` in the sum *is* the reason: "because you watched Start-Up."
  That is precisely the `reason` field in §2.3 of the brief, and it is the
  first-principles interpretability track — derived from the algorithm, not
  narrated by an LLM afterwards.
- Best coverage/quality trade-off measured: 0.95 catalogue coverage at the top
  of the NDCG table.
- Degrades gracefully: a user with one interaction still gets a ranking.

**This is the recommended default personalized algorithm.**

### 3.3 User-based kNN

Same construction transposed. Scored marginally highest on HR@10 (within noise).
Weaker choice in practice: neighbourhoods must be recomputed as users act,
coverage was lowest of the neighbourhood methods (0.57), and the explanation
it yields — "users like you watched this" — is vaguer than an item-level one.
Worth keeping as a registered algorithm for comparison; not the default.

### 3.4 Matrix factorization — SVD / implicit ALS

- `TruncatedSVD` on the binary matrix, and a proper implicit-feedback ALS with
  confidence weighting, both landed at or below random here (§1.3). Expected:
  factorization exists to find latent structure, and there is none to find.
- Costs: a training step, a persisted model, a rank hyperparameter, cold-start
  handling for new users, and latent factors that are not human-readable —
  every explanation must be reconstructed post-hoc.
- Verdict: **worth implementing as a second registered algorithm precisely
  because it is architecturally the hardest case.** An abstraction that can host
  both a zero-training neighbourhood model and a fit/persist/load factorization
  model is proven to be the right abstraction. Not the default.

### 3.5 Content-based (genre + content_type)

User profile = aggregate of the feature vectors of watched items; score by
cosine to candidate items.

- Only 12 usable features exist (8 genres + 4 content types). `title` is free
  text and could be embedded, but the brief explicitly rules out vector
  databases and the titles are real K-drama names whose semantics would come
  from the LLM's world knowledge, not from this dataset.
- Scored below random here — because per-user genre preference is
  indistinguishable from chance (§1.2).
- Nevertheless **structurally valuable**: it is the only family that works for a
  genuinely cold user (given any single interaction or a stated preference), it
  is the mechanism behind the §2.3 `content_type` / `genre` filter, and its
  explanation is trivially honest: "romance, like 6 of your last 10."
- Verdict: implement as the **cold-start and filter layer**, and as a hybrid
  component, not as the standalone default.

### 3.6 Sequential / session-based (Markov chains, GRU4Rec, SASRec)

Ruled out. No sessions exist (median inter-event gap 41 hours, 2 % under an
hour), and a user's own history predicts their next genre *worse* than the
global mode (§1.2). These models also need one to three orders of magnitude
more interactions than the 5,000 available.

### 3.7 Learning-to-rank / two-tower / neural CF

Ruled out on data volume alone. 5,000 interactions across 220 users cannot fit a
neural ranker without memorizing; there are no negative-sampling-proof holdouts
at this size, and §1.3 shows there is no signal for a larger-capacity model to
capture. Reconsider only against a real interaction log.

### 3.8 Hybrid

`α · normalized_personalized + β · normalized_popularity`, with content-based
filtering applied as a post-filter.

Practical value here is not accuracy but **behaviour control**: it guarantees a
full k results for a thin-history user, it is how the fallback stops being a
hard cliff (a user with 2 events gets a blend, not a binary switch to popular),
and α is a single tunable knob. Recommend building it as a composition wrapper
over registered algorithms rather than as an algorithm in its own right.

---

## 4. Libraries — and whether the dependency is worth it

| library | what it gives | assessment for this data |
|---|---|---|
| numpy / scipy only | dense 220×260 matrix, cosine, argsort | **sufficient for §3.1, §3.2, §3.3, §3.5, §3.8.** Zero heavyweight deps. |
| scikit-learn | `TruncatedSVD`, `NearestNeighbors`, `cosine_similarity`, one-hot | worth it for §3.4 and for one-hot feature building; already a normal dependency |
| `implicit` (Frey/ALS, BPR) | fast Cython implicit-feedback ALS/BPR | only if a second factorization algorithm is wanted; the pure-numpy ALS ran in seconds at this size |
| `LightFM` | hybrid factorization with side features (age, region, genre) | the one library whose *design* matches this schema — but §1.2 shows the side features carry no signal, so it would add a dependency to model nothing. Note as a future path. |
| `surprise` | classic explicit-rating CF (SVD, KNNBaseline) | built for explicit star ratings; this is implicit feedback. Poor fit. |
| RecBole / Merlin | dozens of SOTA models, benchmark harness | far past the scale of a 5,000-row fixture |

Recommendation: **numpy + scikit-learn, nothing else.** Every algorithm worth
registering (§3) fits in that, and the small dependency surface keeps the
"clean start, one or two commands" non-functional requirement honest.

---

## 5. Where the LLM sits (Task_2)

Task_2 asks for a short generated reason per personalized recommendation, in
structured JSON, with graceful failure and env-var keys.

The important boundary: **the LLM must explain a decision it did not make.**
It should be handed the evidence the recommender already produced — the
contributing neighbour items from §3.2, the user's genre mix, the popularity
rank — and asked to phrase it. It must never be given the freedom to pick or
reorder items, or the recommender becomes unfalsifiable and the §3.2 native
explanation and the generated one can disagree.

That evidence payload is the same object the first-principles interpretability
track renders directly. One structure, two presentations: the algorithm's own
reason, and the LLM's phrasing of it. When the LLM call fails, the algorithm's
own reason is the fallback text — the degradation is already designed.

---

## 6. Summary recommendation

| slot | choice | reason |
|---|---|---|
| `recommend_popular` | event-type-weighted count + recency decay + Bayesian shrinkage | uses the only informative column; every term explainable |
| `recommend_for_user` default | item-based kNN cosine, top-N truncated neighbours | no training step, native per-item explanation, best coverage |
| second registered algorithm | TruncatedSVD / implicit ALS | proves the abstraction can host a fit/persist model |
| cold start & filtering | content-based over genre + content_type | only family that works with ~zero history; drives the genre filter |
| composition | hybrid blend wrapper | removes the hard fallback cliff, single tunable knob |
| ruled out | sequential, neural CF, LTR | no sessions, no signal, insufficient volume |
| libraries | numpy + scikit-learn | covers all of the above |
| evaluation | leave-last-out harness + bootstrap, run across registered algorithms | the switch-test for the plug-and-play design; also the honest report that this fixture has no signal |

The single most important consequence of §1: **do not tune for accuracy on this
dataset.** Optimize for a clean algorithm boundary, honest explanations, and
correct degradation — and ship the harness that will find the signal on the day
real data arrives.
