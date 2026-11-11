// Offline simulation and shared training use the browser's actual feature and update code.
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { parseArgs } from "node:util";
import { emptyPreferenceProfile, learnPreference, preferenceFeatures, preferenceScore, preferenceFeatureKeys } from "../src/lib/search/preference-model.ts";
import { option } from "../src/test/fixtures.ts";

const { values } = parseArgs({ options: { input: { type: "string" }, output: { type: "string" }, seed: { type: "string", default: "7421" } } });
let seed = Number(values.seed) >>> 0;
const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);
const baseline = JSON.parse(readFileSync(new URL("../src/lib/search/model-release.json", import.meta.url), "utf8"));
const makeOption = (id) => {
  const nonstop = random() < .35;
  const duration = nonstop ? 130 + Math.floor(random() * 50) : 260 + Math.floor(random() * 230);
  const departure = new Date(Date.UTC(2000, 0, 1, 5 + Math.floor(random() * 15)));
  return option({ id, base_price: String(90 + Math.floor(random() * 450)), total_journey_minutes: duration,
    is_nonstop: nonstop, is_self_transfer: !nonstop, connection_minutes: nonstop ? null : duration - 160,
    ticketing_type: nonstop ? "single_ticket" : "separate_tickets", departure_at: departure.toISOString(),
    arrival_at: new Date(+departure + duration * 60_000).toISOString() });
};
// Independent labels based on raw price, duration and clock time; 8% label noise.
function utility(flight, persona, priceSensitivity) {
  const fare = Number(flight.base_price) * priceSensitivity;
  if (persona === "fare") return -fare - .18 * flight.total_journey_minutes;
  if (persona === "time") return -.3 * fare - 1.8 * flight.total_journey_minutes;
  return -.22 * fare - 14 * Math.abs(Number(flight.departure_at.slice(11, 13)) - 10) - (flight.is_self_transfer ? 130 : 0);
}
let records;
if (values.input) {
  records = JSON.parse(readFileSync(values.input, "utf8"));
  if (records.some((row) => row.origin !== "human")) throw new Error("Human training only accepts human-origin records");
} else {
  records = [];
  for (let user = 0; user < 90; user++) {
    const persona = ["fare", "time", "convenience"][user % 3];
    const priceSensitivity = .7 + random() * .6;
    for (let pair = 0; pair < 16; pair++) {
      const left = makeOption(`${user}-${pair}-a`), right = makeOption(`${user}-${pair}-b`);
      let [chosen, rejected] = utility(left, persona, priceSensitivity) >= utility(right, persona, priceSensitivity) ? [left, right] : [right, left];
      const noisy = random() < .08;
      if (noisy) [chosen, rejected] = [rejected, chosen];
      records.push({ event_id: `sim-${values.seed}-${user}-${pair}`, participant: `sim-${String(user).padStart(3, "0")}`, persona, chosen, rejected, noisy, origin: "simulated" });
    }
  }
}
// Participant-level holdout: no traveller appears in more than one split.
const participants = [...new Set(records.map((row) => row.participant))].sort();
if (participants.length < 6 || records.length < 60) throw new Error("Need at least 60 choices from six participants for disjoint train/validation/test groups");
const split = new Map(participants.map((user, i) => [user, i < Math.floor(participants.length * .6) ? "train" : i < Math.floor(participants.length * .8) ? "validation" : "test"]));
records = records.map((row) => ({ ...row, chosen: option({ ...row.chosen, id: `${row.event_id}-chosen` }), rejected: option({ ...row.rejected, id: `${row.event_id}-rejected` }), chosen_features: preferenceFeatures(row.chosen), rejected_features: preferenceFeatures(row.rejected), split: split.get(row.participant) }));
for (const row of records) {
  for (const item of [row.chosen, row.rejected]) {
    const features = preferenceFeatures(item);
    if (preferenceFeatureKeys.some((key) => !Number.isFinite(features[key]))) throw new Error("Invalid training features");
  }
}
const training = records.filter((row) => row.split === "train");
const agreement = (rows, weights) => rows.length ? rows.filter((row) => preferenceScore(row.chosen, weights) > preferenceScore(row.rejected, weights)).length / rows.length : 0;
const candidates = [.03, .1, .3].map((rate) => {
  let profile = { ...emptyPreferenceProfile(), weights: { ...baseline.weights } };
  for (const row of training) profile = learnPreference(profile, row.chosen, row.rejected, rate);
  return { rate, weights: profile.weights, validation: agreement(records.filter((row) => row.split === "validation"), profile.weights) };
});
const selected = candidates.reduce((best, item) => item.validation > best.validation ? item : best);
const evaluate = (weights) => {
  const test = records.filter((row) => row.split === "test");
  return { overall: agreement(test, weights), cohorts: Object.fromEntries([...new Set(test.map((row) => row.persona ?? "human"))].map((persona) => [persona, agreement(test.filter((row) => (row.persona ?? "human") === persona), weights)])) };
};
const reference = evaluate(baseline.weights), candidate = evaluate(selected.weights);
const gatePassed = candidate.overall >= reference.overall + .02 && Object.keys(reference.cohorts).every((cohort) => candidate.cohorts[cohort] >= reference.cohorts[cohort] - .08);
const badWeights = Object.fromEntries(preferenceFeatureKeys.map((key) => [key, -baseline.weights[key]]));
const report = { schemaVersion: 1, seed: values.seed, origin: values.input ? "human" : "simulated", sourceSha256: createHash("sha256").update(readFileSync(new URL("../src/lib/search/preference-model.ts", import.meta.url), "utf8").replaceAll("\r\n", "\n")).digest("hex"),
  participants: participants.length, records, featureKeys: preferenceFeatureKeys, candidates, selected, reference, candidate, bad: evaluate(badWeights), badWeights,
  gatePassed, deploymentEligible: Boolean(values.input && gatePassed), baselineVersion: baseline.version, baselineName: baseline.name,
  predictionExamples: records.filter((row) => row.split === "test").slice(0, 3).map((row) => ({ features: preferenceFeatures(row.chosen), score: preferenceScore(row.chosen, selected.weights) })) };
if (!values.output) throw new Error("Specify --output report.json");
writeFileSync(values.output, JSON.stringify(report));
console.log(JSON.stringify({ origin: report.origin, choices: records.length, participants: participants.length, baseline: reference, candidate, gatePassed, deploymentEligible: report.deploymentEligible }));
