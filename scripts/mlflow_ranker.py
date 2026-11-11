"""Portable MLflow wrapper for Flightly's bounded-feature linear ranker."""

import json
import math

import mlflow


class FlightlyRanker(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["release"], encoding="utf-8") as source:
            self.release = json.load(source)

    def predict(self, context, model_input, params=None):
        keys = self.release["featureKeys"]
        weights = self.release["weights"]
        scores = []
        for row in model_input[keys].to_dict(orient="records"):
            values = [float(row[key]) for key in keys]
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError(
                    "Flightly features must be finite numbers between 0 and 1"
                )
            scores.append(sum(row[key] * weights[key] for key in keys))
        return scores


mlflow.models.set_model(FlightlyRanker())
