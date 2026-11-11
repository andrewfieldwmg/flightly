# /// script
# requires-python = ">=3.12"
# dependencies = ["mlflow==3.18.0"]
# ///
"""Inspect synthetic Flightly preference experiments in local MLflow."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "flightly-mlflow"
EXPERIMENT = "Flightly preference model"
MODEL_NAME = "FlightlyPreferenceRanker"
GUI = "https://desktop-u6i86iq.tail53b14b.ts.net"
SCORER = ROOT / "web/src/lib/search/preference-model.ts"
RELEASE = ROOT / "web/src/lib/search/model-release.json"


def source_hash() -> str:
    return hashlib.sha256(SCORER.read_text(encoding="utf-8").encode()).hexdigest()


def log_model(report: dict, run_id: str) -> None:
    result = next(
        item for item in report["results"] if item["persona"] == "convenience-first"
    )
    release = {
        "schemaVersion": 1,
        "name": MODEL_NAME,
        "modelProfileVersion": report["modelProfileVersion"],
        "featureKeys": report["featureKeys"],
        "sourceSha256": report["sourceSha256"],
        "defaultPersona": result["persona"],
        "dataset": report["dataset"],
        "weights": result["learnedWeights"],
        "trainingBudget": report["trainingBudget"],
        "benchmarkVersion": report["benchmarkVersion"],
        "validationPassed": report["passed"],
        "sourceCommit": report["commit"],
        "predictionExamples": result["predictionExamples"],
        "evaluation": [
            {
                key: item[key]
                for key in ("persona", "pairwiseAccuracy", "minimumAccuracy", "passed")
            }
            for item in report["results"]
        ],
    }
    frame = pd.DataFrame(
        [item["features"] for item in result["predictionExamples"]]
    ).astype(float)
    expected = [item["score"] for item in result["predictionExamples"]]
    with tempfile.TemporaryDirectory() as directory:
        payload = Path(directory) / "release.json"
        payload.write_text(json.dumps(release, indent=2), encoding="utf-8")
        info = mlflow.pyfunc.log_model(
            name="flightly-ranker",
            python_model=str(ROOT / "scripts/mlflow_ranker.py"),
            artifacts={"release": str(payload), "feature_code": str(SCORER)},
            input_example=frame,
            signature=mlflow.models.infer_signature(frame, expected),
            pip_requirements=["mlflow==3.18.0"],
            registered_model_name=MODEL_NAME,
        )
    model = mlflow.pyfunc.load_model(info.model_uri)
    actual = model.predict(frame)
    if not all(
        math.isclose(left, right, abs_tol=1e-10)
        for left, right in zip(actual, expected, strict=True)
    ):
        raise RuntimeError("Packaged model predictions differ from the browser ranker")
    mlflow.log_metrics(
        {
            "pairwise_agreement": result["pairwiseAccuracy"],
            "regression_floor": result["minimumAccuracy"],
            "cheapest_first_baseline": result["cheapestFirstAccuracy"],
        },
        run_id=run_id,
        model_id=info.model_id,
    )
    client = mlflow.MlflowClient()
    version = str(info.registered_model_version)
    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "validation",
        "passed" if report["passed"] else "below_floor",
    )
    client.set_model_version_tag(MODEL_NAME, version, "prediction_parity", "passed")
    client.set_model_version_tag(
        MODEL_NAME, version, "training_budget", str(report["trainingBudget"])
    )
    client.update_model_version(
        MODEL_NAME,
        version,
        description=f"Synthetic convenience-first prior from {report['trainingBudget']} choices. "
        f"Browser profile v{report['modelProfileVersion']}. Source run {run_id}. "
        "Personal browser learning takes precedence over these initial weights.",
    )
    print(
        f"Registered {MODEL_NAME} v{version}: {GUI}/#/models/{MODEL_NAME}/versions/{version}"
    )


def publish(version: str) -> None:
    client = mlflow.MlflowClient()
    info = client.get_model_version(MODEL_NAME, version)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    release = model.unwrap_python_model().release
    if (
        not release["validationPassed"]
        or info.tags.get("prediction_parity") != "passed"
    ):
        raise ValueError(
            "Only validated models with matching browser predictions can be published"
        )
    if source_hash() != release["sourceSha256"]:
        raise ValueError(
            "Browser scorer has changed since training; record a fresh run first"
        )
    frame = pd.DataFrame(
        [item["features"] for item in release["predictionExamples"]]
    ).astype(float)
    actual = model.predict(frame)
    expected = [item["score"] for item in release["predictionExamples"]]
    if not all(
        math.isclose(a, b, abs_tol=1e-10) for a, b in zip(actual, expected, strict=True)
    ):
        raise ValueError("Registry model failed its recorded prediction examples")
    release.update(
        {
            "version": version,
            "modelId": info.source.removeprefix("models:/"),
            "runId": info.run_id,
            "registryUrl": f"{GUI}/#/models/{MODEL_NAME}/versions/{version}",
        }
    )
    mlflow.log_metrics(
        {
            "pairwise_agreement": next(
                item["pairwiseAccuracy"]
                for item in release["evaluation"]
                if item["persona"] == release["defaultPersona"]
            ),
            "regression_floor": next(
                item["minimumAccuracy"]
                for item in release["evaluation"]
                if item["persona"] == release["defaultPersona"]
            ),
        },
        run_id=info.run_id,
        model_id=release["modelId"],
    )
    temporary = RELEASE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    temporary.replace(RELEASE)
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    client.set_model_version_tag(
        MODEL_NAME, version, "release_status", "exported_for_deployment"
    )
    print(f"Exported registry v{version} to {RELEASE}")


def serve(port: int) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
            "--backend-store-uri",
            "sqlite:///tracking.db",
            "--artifacts-destination",
            str(DATA / "artifacts"),
        ],
        cwd=DATA,
    )


def record(budgets: list[int], port: int) -> None:
    uri = f"http://127.0.0.1:{port}"
    # The script prints its own links and works in non-UTF-8 Windows terminals.
    os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.set_experiment(EXPERIMENT)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    env = {**os.environ, "GITHUB_SHA": commit}
    env.pop("GITHUB_STEP_SUMMARY", None)

    for budget in budgets:
        completed = subprocess.run(
            [
                "node",
                "scripts/evaluate-preferences.mjs",
                "--training-budget",
                str(budget),
            ],
            cwd=ROOT / "web",
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        # Exit 1 with a complete report is an informative below-floor result.
        if completed.returncode not in (0, 1) or not completed.stdout:
            raise RuntimeError(
                completed.stderr or "The evaluator did not produce a report"
            )
        report = json.loads(completed.stdout)
        with mlflow.start_run(run_name=f"{budget} training choices") as run:
            mlflow.set_tags(
                {
                    "mlflow.source.git.commit": commit,
                    "source_tree": "dirty" if dirty else "clean",
                    "run_type": "CI reference" if budget == 36 else "budget experiment",
                    "regression_floors": "passed"
                    if report["passed"]
                    else "below floor",
                }
            )
            mlflow.log_params(
                {
                    "training_budget": budget,
                    "benchmark_version": report["benchmarkVersion"],
                    "model_profile_version": report["modelProfileVersion"],
                    "dataset": report["dataset"],
                    "training_seed": report["seeds"]["training"],
                    "evaluation_seed": report["seeds"]["evaluation"],
                    "options_per_split": report["optionsPerSplit"],
                }
            )
            metrics = {}
            for result in report["results"]:
                persona = result["persona"].replace("-", "_")
                for suffix, field in (
                    ("agreement", "pairwiseAccuracy"),
                    ("cheapest_first", "cheapestFirstAccuracy"),
                    ("floor", "minimumAccuracy"),
                    ("choices", "trainingComparisons"),
                ):
                    if result[field] is not None:
                        metrics[f"{persona}_{suffix}"] = result[field]
            mlflow.log_metrics(metrics)
            mlflow.log_dict(report, "preference-evaluation.json")
            log_model(report, run.info.run_id)
        print(
            f"{budget} choices: {uri}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
        )

    print(
        f"Compare runs: {uri}/#/experiments/{experiment.experiment_id}/runs"
        "?workflowType=machine_learning"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve", help="start the local MLflow GUI")
    runs = commands.add_parser("run", help="record one or more training budgets")
    release = commands.add_parser(
        "publish", help="export a validated registry version for the app"
    )
    release.add_argument("version")
    for command in (server, runs, release):
        command.add_argument("--port", type=int, default=5505)
    runs.add_argument("budgets", type=int, nargs="*", default=[12, 36, 72])
    args = parser.parse_args()
    if args.command == "serve":
        return serve(args.port)
    if args.command == "publish":
        os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
        mlflow.set_tracking_uri(f"http://127.0.0.1:{args.port}")
        publish(args.version)
        return 0
    if any(budget < 1 or budget > 120 for budget in args.budgets):
        runs.error("training budgets must be between 1 and 120")
    record(args.budgets, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
