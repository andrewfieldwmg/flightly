import { writeFileSync } from "node:fs";
import {
  choosePreferencePair,
  emptyPreferenceProfile,
  learnPreference,
  PREFERENCE_MODEL_VERSION,
  preferenceScore,
} from "../src/lib/search/preference-model.ts";
import { option } from "../src/test/fixtures.ts";

const benchmarkVersion = 1;
const trainingBudget = 36;
const optionsPerSplit = 18;
const seeds = { training: 5281, evaluation: 9137 };

function makeOptions(split) {
  let state = seeds[split];
  const draw = (size) => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return Math.floor(state / 2 ** 32 * size);
  };
  return Array.from({ length: optionsPerSplit }, (_, index) => {
    const nonstop = index % 3 === 0;
    const separate = !nonstop && index % 3 === 1;
    const duration = nonstop ? 135 + draw(51) : 235 + draw(251);
    const hour = 5 + draw(14);
    const departure = new Date(Date.UTC(2026, 11, 18, hour));
    return option({
      id: `${split}-${index}`,
      base_price: String(95 + draw(446)),
      total_journey_minutes: duration,
      departure_at: departure.toISOString(),
      arrival_at: new Date(departure.getTime() + duration * 60_000).toISOString(),
      is_nonstop: nonstop,
      is_self_transfer: separate,
      connection_minutes: nonstop ? null : duration - 145,
      ticketing_type: separate ? "separate_tickets" : "single_ticket",
    });
  });
}

const training = makeOptions("training");
const evaluation = makeOptions("evaluation");
const fare = (flight) => Number(flight.base_price);
const departureHour = (flight) => Number(flight.departure_at.slice(11, 13));

// Labels are generated in pounds from raw itinerary fields, independently
// of the production feature extraction and scoring functions.
const personas = {
  "fare-first": {
    minimumAccuracy: 0.50,
    labelScore: (flight) => -fare(flight) - 0.18 * flight.total_journey_minutes,
  },
  "time-first": {
    minimumAccuracy: 0.80,
    labelScore: (flight) => -0.3 * fare(flight) - 1.8 * flight.total_journey_minutes,
  },
  "convenience-first": {
    minimumAccuracy: 0.85,
    labelScore: (flight) => (
      -0.22 * fare(flight)
      - 14 * Math.abs(departureHour(flight) - 10)
      - (flight.is_self_transfer ? 130 : 0)
    ),
  },
};

function benchmarkPersona(name, { labelScore, minimumAccuracy }) {
  let profile = emptyPreferenceProfile();
  for (let index = 0; index < trainingBudget; index += 1) {
    const pair = choosePreferencePair(training, profile);
    if (!pair) break;
    const { left, right } = pair;
    const [chosen, rejected] = labelScore(left) >= labelScore(right) ? [left, right] : [right, left];
    profile = learnPreference(profile, chosen, rejected);
  }

  const scores = evaluation.map((flight) => preferenceScore(flight, profile.weights));
  let correct = 0;
  let cheapestCorrect = 0;
  let labelledPairs = 0;
  for (let left = 0; left < evaluation.length; left += 1) {
    for (let right = left + 1; right < evaluation.length; right += 1) {
      const expected = Math.sign(labelScore(evaluation[left]) - labelScore(evaluation[right]));
      if (expected === 0) continue;
      labelledPairs += 1;
      if (Math.sign(scores[left] - scores[right]) === expected) correct += 1;
      if (Math.sign(fare(evaluation[right]) - fare(evaluation[left])) === expected) cheapestCorrect += 1;
    }
  }

  const accuracy = labelledPairs ? correct / labelledPairs : null;
  return {
    persona: name,
    trainingComparisons: profile.comparisons,
    labelledPairs,
    correct,
    pairwiseAccuracy: accuracy,
    cheapestFirstAccuracy: labelledPairs ? cheapestCorrect / labelledPairs : null,
    minimumAccuracy,
    passed: profile.comparisons === trainingBudget
      && scores.every(Number.isFinite)
      && accuracy !== null && accuracy >= minimumAccuracy,
  };
}

const results = Object.entries(personas).map(([name, persona]) => benchmarkPersona(name, persona));
const report = {
  benchmarkVersion,
  modelProfileVersion: PREFERENCE_MODEL_VERSION,
  commit: process.env.GITHUB_SHA ?? null,
  dataset: "synthetic",
  seeds,
  optionsPerSplit,
  trainingBudget,
  results,
  passed: results.every((result) => result.passed),
};

const json = JSON.stringify(report, null, 2) + "\n";
if (process.argv[2]) writeFileSync(process.argv[2], json);
process.stdout.write(json);
if (!report.passed) process.exitCode = 1;
