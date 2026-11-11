import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { preferenceFeatureKeys, PREFERENCE_MODEL_VERSION } from "../src/lib/search/preference-model.ts";

const release = JSON.parse(readFileSync(new URL("../src/lib/search/model-release.json", import.meta.url), "utf8"));
const source = readFileSync(new URL("../src/lib/search/preference-model.ts", import.meta.url), "utf8").replaceAll("\r\n", "\n");
assert.equal(release.schemaVersion, 1);
assert.ok(["FlightlyPreferenceRanker", "FlightlySharedPreferenceRanker"].includes(release.name));
if (release.name === "FlightlySharedPreferenceRanker") {
  assert.equal(release.origin, "human");
  assert.equal(release.deploymentEligible, true);
  assert.equal(release.dataset, "human");
}
assert.match(release.version, /^\d+$/);
assert.equal(release.validationPassed, true);
assert.equal(release.modelProfileVersion, PREFERENCE_MODEL_VERSION);
assert.deepEqual(release.featureKeys, preferenceFeatureKeys);
assert.equal(release.sourceSha256, createHash("sha256").update(source).digest("hex"), "Scorer differs from registry model. Train and publish a matching model before building.");
for (const key of preferenceFeatureKeys) assert.ok(Number.isFinite(release.weights[key]));
assert.ok(release.evaluation.length > 0);
for (const result of release.evaluation) {
  assert.equal(result.passed, true);
  assert.ok(result.pairwiseAccuracy >= result.minimumAccuracy);
}
assert.ok(release.predictionExamples.length > 0);
for (const example of release.predictionExamples) {
  const score = preferenceFeatureKeys.reduce((sum, key) => sum + example.features[key] * release.weights[key], 0);
  assert.ok(Math.abs(score - example.score) < 1e-10, "Exported weights differ from packaged predictions");
}
console.log(`Verified ${release.name} v${release.version} against its registry feature contract and predictions.`);
