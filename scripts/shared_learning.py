# /// script
# requires-python = ">=3.12"
# dependencies = ["mlflow==3.18.0"]
# ///
"""Flightly learning lab and private feedback sync, with no paid provider calls."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from urllib.request import Request, urlopen

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")

import mlflow
import pandas as pd
from mlflow.genai.datasets import create_dataset, search_datasets
from mlflow.genai.scorers import scorer

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/flightly-mlflow/shared"
GUI = "https://desktop-u6i86iq.tail53b14b.ts.net"
MODEL = "FlightlySharedPreferenceRanker"
BASELINE = ROOT / "web/src/lib/search/model-release.json"


def dataset(name, experiment, origin):
    matches = search_datasets(
        experiment_ids=[experiment], filter_string=f"name = '{name}'", max_results=1
    )
    return (
        matches[0]
        if matches
        else create_dataset(
            name=name,
            experiment_id=[experiment],
            tags={"origin": origin, "contains_pii": "false"},
        )
    )


def records_for_mlflow(report, split):
    return [
        {
            "inputs": {
                "event_id": row["event_id"],
                "chosen": row["chosen_features"],
                "rejected": row["rejected_features"],
            },
            "expectations": {
                "chosen_wins": True,
                "cohort": row.get("persona", "human"),
                "origin": report["origin"],
            },
        }
        for row in report["records"]
        if row["split"] == split
    ]


def predict_pair(weights, event_id, chosen, rejected):
    with mlflow.start_span(
        name="linear_pairwise_prediction", span_type="RANKER"
    ) as span:
        span.set_inputs({"event_id": event_id, "chosen": chosen, "rejected": rejected})
        chosen_score = sum(chosen[key] * weights[key] for key in weights)
        rejected_score = sum(rejected[key] * weights[key] for key in weights)
        result = {
            "chosen_wins": chosen_score > rejected_score,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "margin": chosen_score - rejected_score,
            "contributions": {
                key: (chosen[key] - rejected[key]) * weights[key] for key in weights
            },
        }
        span.set_outputs(result)
        return result


@scorer
def preference_agreement(outputs, expectations) -> bool:
    return outputs["chosen_wins"] == expectations["chosen_wins"]


def package(report, weights, label, passed):
    release = {
        "schemaVersion": 1,
        "name": MODEL,
        "modelProfileVersion": 1,
        "featureKeys": report["featureKeys"],
        "weights": weights,
        "sourceSha256": report["sourceSha256"],
        "origin": report["origin"],
        "validationPassed": passed,
        "deploymentEligible": report["deploymentEligible"] and passed,
        "defaultPersona": "shared-human"
        if report["origin"] == "human"
        else "shared-simulated",
        "dataset": report["origin"],
        "trainingBudget": sum(row["split"] == "train" for row in report["records"]),
        "benchmarkVersion": 2,
        "baselineVersion": report["baselineVersion"],
        "baselineName": report["baselineName"],
        "predictionExamples": [
            {
                "features": item["features"],
                "score": sum(item["features"][key] * weights[key] for key in weights),
            }
            for item in report["predictionExamples"]
        ],
        "evaluation": [
            {
                "persona": "shared-human",
                "pairwiseAccuracy": report["candidate"]["overall"],
                "minimumAccuracy": report["reference"]["overall"] + 0.02,
                "passed": passed,
            },
            *[
                {
                    "persona": cohort,
                    "pairwiseAccuracy": score,
                    "minimumAccuracy": report["reference"]["cohorts"][cohort] - 0.08,
                    "passed": passed,
                }
                for cohort, score in report["candidate"]["cohorts"].items()
            ],
        ],
    }
    frame = pd.DataFrame(
        [example["features"] for example in report["predictionExamples"]]
    ).astype(float)
    expected = [
        sum(row[key] * weights[key] for key in weights)
        for row in frame.to_dict("records")
    ]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "release.json"
        path.write_text(json.dumps(release), encoding="utf-8")
        info = mlflow.pyfunc.log_model(
            name=label,
            python_model=str(ROOT / "scripts/mlflow_ranker.py"),
            artifacts={
                "release": str(path),
                "feature_code": str(ROOT / "web/src/lib/search/preference-model.ts"),
            },
            input_example=frame,
            signature=mlflow.models.infer_signature(frame, expected),
            pip_requirements=["mlflow==3.18.0"],
            registered_model_name=MODEL,
        )
    actual = mlflow.pyfunc.load_model(info.model_uri).predict(frame)
    if any(abs(a - b) > 1e-10 for a, b in zip(actual, expected, strict=True)):
        raise ValueError("Packaged shared model differs from browser scores")
    client = mlflow.MlflowClient()
    for key, value in {
        "origin": report["origin"],
        "validation": "passed" if passed else "rejected",
        "deployment_eligible": str(release["deploymentEligible"]).lower(),
        "prediction_parity": "passed",
    }.items():
        client.set_model_version_tag(MODEL, info.registered_model_version, key, value)
    if label == "shared-candidate":
        client.set_registered_model_alias(
            MODEL, "shared-challenger", str(info.registered_model_version)
        )
    return info


def lab(human_input=None):
    WORK.mkdir(parents=True, exist_ok=True)
    report_path = WORK / (
        "human-report.json" if human_input else "simulation-report.json"
    )
    command = ["node", "scripts/shared-learning.mjs", "--output", str(report_path)]
    if human_input:
        command.extend(["--input", str(human_input)])
    subprocess.run(command, cwd=ROOT / "web", check=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    origin = report["origin"]
    experiment = mlflow.set_experiment(f"Flightly shared learning - {origin}")
    eid = experiment.experiment_id
    datasets = {}
    for split in ("train", "validation", "test"):
        data = dataset(f"flightly-{origin}-{split}-v1", eid, origin)
        data.merge_records(records_for_mlflow(report, split))
        datasets[split] = data
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    ids = []
    with mlflow.start_run(
        run_name=f"Shared learning - {origin} - {report['participants']} travellers"
    ) as parent:
        mlflow.set_tags(
            {
                "origin": origin,
                "split_policy": "disjoint_participants",
                "production_changed": "false",
                "mlflow.source.git.commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "source_tree": "dirty"
                if subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
                else "clean",
            }
        )
        mlflow.log_artifact(str(report_path), "reproducibility")
        mlflow.log_artifacts(str(ROOT / "web/scripts"), "evaluation-source")
        mlflow.log_params(
            {
                "choices": len(report["records"]),
                "travellers": report["participants"],
                "seed": report["seed"],
                "baseline_version": report["baselineVersion"],
                "label_noise": 0.08 if origin == "simulated" else "unknown",
                "test_tuning": "never",
                "promotion_policy": "2pp uplift; max 8pp cohort regression; human-only deployment",
            }
        )
        for rate in report["candidates"]:
            with mlflow.start_run(
                run_name=f"Tune learning rate {rate['rate']}", nested=True
            ):
                mlflow.log_param("learning_rate", rate["rate"])
                mlflow.log_metric("validation_agreement", rate["validation"])
        for label, weights, result, passed in (
            ("production-baseline", baseline["weights"], report["reference"], True),
            (
                "shared-candidate",
                report["selected"]["weights"],
                report["candidate"],
                report["gatePassed"],
            ),
            ("inverted-weights-failure", report["badWeights"], report["bad"], False),
        ):
            with mlflow.start_run(run_name=label, nested=True) as run:
                mlflow.set_tags(
                    {
                        "origin": origin,
                        "validation": "passed" if passed else "rejected",
                        "failure_injection": str(
                            label == "inverted-weights-failure"
                        ).lower(),
                    }
                )
                mlflow.log_params(
                    {
                        "learning_rate": report["selected"]["rate"]
                        if label == "shared-candidate"
                        else "unchanged",
                        "origin": origin,
                    }
                )
                mlflow.log_metrics(
                    {
                        "test_agreement": result["overall"],
                        "uplift_vs_baseline": result["overall"]
                        - report["reference"]["overall"],
                        **{
                            f"cohort_{key}": value
                            for key, value in result["cohorts"].items()
                        },
                    }
                )
                for split in ("train", "validation", "test"):
                    rows = [row for row in report["records"] if row["split"] == split]
                    tracking = mlflow.data.from_pandas(
                        pd.DataFrame(
                            [
                                {
                                    "event_id": row["event_id"],
                                    "participant": row["participant"],
                                    "origin": origin,
                                    "chosen_wins": True,
                                }
                                for row in rows
                            ]
                        ),
                        source=str(report_path),
                        name=f"flightly-{origin}-{split}",
                        targets="chosen_wins",
                    )
                    mlflow.log_input(tracking, context=split)
                mlflow.log_dict(
                    {
                        "weights": weights,
                        "results": result,
                        "gate_passed": passed,
                        "deployment_eligible": report["deploymentEligible"] and passed,
                    },
                    "decision.json",
                )
                info = package(report, weights, label, passed)

                # Native evaluation UI: deterministic scorer, no LLM/judge/provider calls.
                evaluation = mlflow.genai.evaluate(
                    data=datasets["test"],
                    predict_fn=partial(predict_pair, weights),
                    scorers=[preference_agreement],
                    model_id=info.model_id,
                )
                ids.append(
                    {
                        "label": label,
                        "run_id": run.info.run_id,
                        "evaluation_run_id": evaluation.run_id,
                        "model_version": info.registered_model_version,
                    }
                )
        # This is an intentionally thrown validation error, distinct from wrong predictions.
        try:
            with mlflow.start_span(
                name="injected_invalid_features",
                attributes={"origin": origin, "failure_injection": True},
            ) as span:
                span.set_inputs(
                    {
                        "affordability": "invalid",
                        "scenario": "demonstrate feature validation failure",
                    }
                )
                invalid_frame = pd.DataFrame(
                    [report["predictionExamples"][0]["features"]]
                ).astype(float)
                invalid_frame["affordability"] = float("nan")
                # Exercise the packaged model's actual validation guard.
                mlflow.pyfunc.load_model(info.model_uri).predict(invalid_frame)
                raise AssertionError("The packaged ranker accepted invalid features")
        except ValueError:
            trace_id = mlflow.get_last_active_trace_id()
            mlflow.log_dict(
                {"trace_id": trace_id, "expected": "ERROR", "injected": True},
                "injected-failure.json",
            )
        mlflow.log_dict(
            {
                "origin": origin,
                "baseline": report["reference"],
                "candidate": report["candidate"],
                "selected_rate": report["selected"]["rate"],
                "gate_passed": report["gatePassed"],
                "deployment_eligible": report["deploymentEligible"],
                "runs": ids,
            },
            "lab-summary.json",
        )
    mlflow.flush_trace_async_logging()
    # Persisted trace readback proves this experiment is connected to the real server.
    trace = mlflow.get_trace(trace_id)
    if trace is None or not trace.data.spans:
        raise RuntimeError(f"Trace not persisted: {trace_id}")
    links = {
        "experiment_id": eid,
        "parent_run_id": parent.info.run_id,
        "trace_id": trace_id,
        "runs": ids,
        "datasets": {key: value.dataset_id for key, value in datasets.items()},
    }
    (WORK / f"{origin}-links.json").write_text(
        json.dumps(links, indent=2), encoding="utf-8"
    )
    print(json.dumps(links))
    print(f"Open {GUI}/#/experiments/{eid}/runs?workflowType=machine_learning")


def sync(config_file, watch):
    WORK.mkdir(parents=True, exist_ok=True)
    state_path = WORK / "sync-state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"seen": [], "trained_count": 0}
    )
    while True:
        try:
            exported = WORK / "human.json"
            config = json.loads(config_file.read_text())
            rows = []
            after = ""
            while True:
                request = Request(
                    f"{config['url']}/api/learning/export?after={after}",
                    headers={"Authorization": f"Bearer {config['token']}"},
                )
                with urlopen(request, timeout=20) as response:
                    page = json.load(response)
                rows.extend(page["records"])
                if not page["next"]:
                    break
                after = page["next"]
            exported.write_text(json.dumps(rows), encoding="utf-8")
            experiment = mlflow.set_experiment("Flightly shared learning - human")
            data = dataset(
                "flightly-human-feedback-v1", experiment.experiment_id, "human"
            )
            # Feature derivation is the exact production TypeScript code, never a Python approximation.
            feature_path = WORK / "human-features.json"
            subprocess.run(
                [
                    "node",
                    "scripts/feedback-features.mjs",
                    str(exported),
                    str(feature_path),
                ],
                cwd=ROOT / "web",
                check=True,
                capture_output=True,
            )
            rows = json.loads(feature_path.read_text())
            seen = set(state["seen"])
            pending = [row for row in rows if row["event_id"] not in seen]
            baseline = json.loads(BASELINE.read_text())
            if pending:
                data.merge_records(
                    [
                        {
                            "inputs": {
                                "event_id": row["event_id"],
                                "chosen": row["chosen_features"],
                                "rejected": row["rejected_features"],
                            },
                            "expectations": {"chosen_wins": True, "origin": "human"},
                        }
                        for row in pending
                    ]
                )
                with mlflow.start_run(
                    run_name=f"Feedback sync - {len(pending)} new human choices"
                ):
                    mlflow.set_tag("origin", "human")
                    mlflow.log_metrics(
                        {
                            "new_choices": len(pending),
                            "total_choices": len(rows),
                            "participants": len({row["participant"] for row in rows}),
                        }
                    )
                    for row in pending:
                        with mlflow.start_span(name="shared_human_choice") as span:
                            mlflow.update_current_trace(
                                tags={
                                    "origin": "human",
                                    "event_id": row["event_id"],
                                    "browser_model_version": row["model_version"],
                                },
                                user=row["participant"],
                            )
                            span.set_inputs(
                                {
                                    "event_id": row["event_id"],
                                    "chosen": row["chosen_features"],
                                    "rejected": row["rejected_features"],
                                }
                            )
                            output = predict_pair(
                                baseline["weights"],
                                row["event_id"],
                                row["chosen_features"],
                                row["rejected_features"],
                            )
                            span.set_outputs(
                                {
                                    **output,
                                    "reference_model_version": baseline["version"],
                                    "meaning": "Batch replay against the current registered prior; browser personal weights are not collected",
                                }
                            )
                        trace_id = mlflow.get_last_active_trace_id()
                        mlflow.log_feedback(
                            trace_id=trace_id,
                            name="prior_agrees_with_human",
                            value=output["chosen_wins"],
                            rationale="Explicit opted-in A/B preference, replayed against the registered prior",
                        )
                        seen.add(row["event_id"])
                mlflow.flush_trace_async_logging()
                state["seen"] = sorted(seen)
                print(f"Synced {len(pending)} new human choices", flush=True)
            if (
                len(rows) >= 60
                and len({row["participant"] for row in rows}) >= 6
                and len(rows) - state["trained_count"] >= 20
            ):
                lab(exported)
                state["trained_count"] = len(rows)
            state.update(
                {
                    "last_sync_at": time.time(),
                    "total_choices": len(rows),
                    "human_dataset_id": data.dataset_id,
                    "experiment_id": experiment.experiment_id,
                }
            )
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(state_path)
        except Exception as error:  # noqa: BLE001 - persistent bridge must retry without leaking secrets
            # DB URLs and subprocess stderr can contain secrets; report class only.
            print(
                f"Feedback sync failed ({type(error).__name__}); retrying next interval",
                file=sys.stderr,
                flush=True,
            )
            if not watch:
                raise RuntimeError(
                    "Feedback sync failed; inspect private service logs"
                ) from None
        if not watch:
            return
        time.sleep(30)


def publish_shared(version):
    """Export a human-validated shared prior; deployment stays an explicit Git release."""
    import hashlib

    client = mlflow.MlflowClient()
    info = client.get_model_version(MODEL, version)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL}/{version}")
    release = model.unwrap_python_model().release
    if (
        release.get("origin") != "human"
        or not release.get("deploymentEligible")
        or not release.get("validationPassed")
        or info.tags.get("prediction_parity") != "passed"
    ):
        raise ValueError(
            "Only human-origin models that pass shared holdout gates can be released"
        )
    source = (ROOT / "web/src/lib/search/preference-model.ts").read_text(
        encoding="utf-8"
    )
    if hashlib.sha256(source.encode()).hexdigest() != release["sourceSha256"]:
        raise ValueError("Scorer changed: train a fresh shared candidate")
    baseline = json.loads(BASELINE.read_text())
    if (
        baseline["version"] != release["baselineVersion"]
        or baseline["name"] != release["baselineName"]
    ):
        raise ValueError(
            "Production prior changed: compare against the current baseline first"
        )
    examples = release["predictionExamples"]
    predictions = model.predict(
        pd.DataFrame([row["features"] for row in examples]).astype(float)
    )
    if any(
        abs(a - b["score"]) > 1e-10 for a, b in zip(predictions, examples, strict=True)
    ):
        raise ValueError("Registry/browser prediction parity failed")
    release.update(
        version=str(version),
        modelId=info.source.removeprefix("models:/"),
        runId=info.run_id,
        registryUrl=f"{GUI}/#/models/{MODEL}/versions/{version}",
    )
    temporary = BASELINE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    temporary.replace(BASELINE)
    client.set_registered_model_alias(MODEL, "champion", str(version))
    print(
        "Exported validated shared prior. Run npm run check:model, then release through Git/Vercel."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["lab", "sync", "publish"])
    parser.add_argument("version", nargs="?")
    parser.add_argument("--human-input", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    mlflow.set_tracking_uri("http://127.0.0.1:5505")
    if args.command == "lab":
        lab(args.human_input)
    elif args.command == "publish" and args.version:
        publish_shared(args.version)
    elif args.command == "sync" and args.config:
        sync(args.config, args.watch)
    else:
        parser.error("sync requires --config")
