# Flightly shared-learning lab

The population model is a cold-start prior for a new browser. Personal A/B choices
then update that browser's weights. This is a linear pairwise logistic ranker;
there are no LLM calls, paid judges or flight-search API calls in this workflow.

## Start in the GUI

Use the existing private [MLflow dashboard](https://desktop-u6i86iq.tail53b14b.ts.net).
Your other machine needs to be signed into the same Tailscale network.

1. Open **Flightly shared learning - simulated**, then **Model training → Runs**.
   Expand the latest parent run. Compare `production-baseline`, `shared-candidate`
   and `inverted-weights-failure`. Compare `test_agreement` and the three cohort
   metrics. A rise in overall agreement can hide a decline for one traveller type.
2. Inspect the three **Tune learning rate** child runs. The rate is selected on
   validation travellers, before touching test travellers. Train/validation/test
   travellers do not overlap. The simulator generates 90 travellers, 16 choices
   each, three different tastes, varying price sensitivity and 8% noisy labels.
3. Switch to **GenAI → Datasets** inside that same experiment. Open the train,
   validation and test datasets. Inputs contain two feature vectors; the
   expectation is that the explicitly chosen option ranks higher.
4. Open **Evaluation runs**. Inspect an example with `preference_agreement = false`.
   Its prediction trace contains both scores, their margin and the contribution
   of every feature. These evaluations use a deterministic scorer, despite being
   displayed in MLflow's GenAI area. No language model is involved.
5. Open **Traces** and find `injected_invalid_features`. This deliberate error
   illustrates a broken request. A normal completed trace with a failed agreement
   score illustrates a valid request with an unwanted recommendation.
6. Open the model registry's **FlightlySharedPreferenceRanker**. The
   `shared-challenger` alias identifies the candidate. Check `origin`,
   `validation`, `prediction_parity` and `deployment_eligible`. Simulation models
   cannot be exported as production shared priors.

## Add a real choice

In Flightly, load an existing cached search with at least two options. Under the
preference trainer, enable **Help improve recommendations for everyone**, then
answer an A/B question. This remains compatible with cached-only searching.

Personal learning happens immediately. The shared label is verified against the
stored search and saved in PostgreSQL. The local worker polls a private,
token-protected export every 30 seconds. Look in **Flightly shared learning -
human**, its feedback dataset and its traces. A replay compares your choice with
the registered prior, not with your browser's evolving personal weights.

Sharing defaults off. No names, routes, travel dates, search IDs or raw browser
identifiers enter the learning dataset. Clock times and ranking inputs are kept;
a pseudonymous participant hash supports traveller-level holdout. Opting out
clears pending uploads; it does not remove previously shared labels. Offline
uploads retry with a stable event ID, so retries do not create duplicate labels.
Device identifiers do not prove distinct real people or prevent malicious voting.

Human and simulated labels live in separate experiments/datasets. The empty
human dataset is intentional until someone opts in. Never describe simulation
metrics as evidence of real-user accuracy or observed business uplift.

## Train, compare and release

The background worker builds a human candidate after at least 60 choices from six
device participants, then after each additional 20 choices. These are lab minimums,
not a claim of statistical sufficiency. It requires 2 percentage points of overall
held-out improvement and allows at most 8 points of cohort regression. A single
user cannot supply cross-user evidence, regardless of choice count. Human labels
have no synthetic persona classifications, so their cohort is reported as `human`.

Candidates enter the registry. They do not silently replace production. To
release a validated human version, export it, check the manifest and deploy:

```powershell
uv run --script scripts/shared_learning.py publish VERSION
cd web
npm run check:model
npm run build
```

Commit the generated `web/src/lib/search/model-release.json` and release through
Git/Vercel. New/reset browsers receive the shared prior. Browsers with personal
learning retain it. The current production model remains registry v4 until a
human-validated shared model is released.

To reproduce the lab or sync once without invoking a flight provider:

```powershell
uv run --script scripts/shared_learning.py lab
uv run --script scripts/shared_learning.py sync --config work/flightly-feedback-sync.json
```

The simulator's seed is fixed. JSON reports, source hashes, model artifacts and
dataset lineage are logged. CI also exports the shared simulation report.
Native evaluation views show per-example scores and traces. In this lab the same
baseline/candidate run also holds aggregate metrics and a model package; it can
appear in both training and evaluation views. The learning-rate trials are
separate child runs. An experiment groups these related runs and datasets.

## Background services

`Flightly-MLflow` keeps the local dashboard running. `Flightly-FeedbackSync`
runs `scripts/start-flightly-feedback.ps1` hidden at sign-in and restarts on
failure. Both require this PC to be on and the Administrator session signed in.
The feedback worker is independent of a coding-agent session.

Logs and state are under ignored `work/flightly-mlflow/shared/`. The private
`work/flightly-feedback-sync.json` token authorizes only sanitized feedback export;
the deployed copy is the sensitive `FLIGHTLY_FEEDBACK_SYNC_TOKEN` variable. No
production database credentials are downloaded. Alembic migrations run inside
production builds under a PostgreSQL advisory lock; preview builds skip them.
If the poller stops, feedback stays in PostgreSQL and is picked up on recovery.
An interrupted sync may replay a trace, but dataset records deduplicate by inputs.

Prompts, LLM gateways and agent tool calls are not part of Flightly. They belong
to the separate RAG/agent applications. MLflow's deterministic evaluation UI is
useful here without adding any of those features.
