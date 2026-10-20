import { describe, expect, it } from "vitest";
import { option } from "@/test/fixtures";
import {
  choosePreferencePair,
  emptyPreferenceProfile,
  learnPreference,
  pairKey,
  parsePreferenceProfile,
  preferenceFeatureKeys,
  preferenceFeatures,
  preferenceScore,
  type PreferenceWeights,
} from "./preference-model";

const direct = option({ id: "direct", base_price: "420", total_journey_minutes: 150, is_nonstop: true });
const cheapConnection = option({
  id: "cheap-connection",
  base_price: "130",
  total_journey_minutes: 390,
  is_nonstop: false,
  is_self_transfer: true,
  connection_airport: "BGY",
  connection_minutes: 120,
  ticketing_type: "separate_tickets",
});

describe("online preference model", () => {
  it("produces finite, bounded, explainable features", () => {
    const features = preferenceFeatures(cheapConnection);
    expect(Object.keys(features)).toEqual(preferenceFeatureKeys);
    for (const value of Object.values(features)) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
    expect(features.affordability).toBeGreaterThan(preferenceFeatures(direct).affordability);
    expect(features.nonstop).toBe(0);
  });

  it("learns a user's revealed trade-off from pairwise choices", () => {
    let profile = emptyPreferenceProfile();
    for (let index = 0; index < 5; index += 1) profile = learnPreference(profile, cheapConnection, direct);

    expect(profile.comparisons).toBe(5);
    expect(preferenceScore(cheapConnection, profile.weights)).toBeGreaterThan(preferenceScore(direct, profile.weights));
    expect(profile.weights.affordability).toBeGreaterThan(0);
    expect(profile.weights.nonstop).toBeLessThan(0);
  });

  it("uses uncertainty sampling and never repeats a seen comparison", () => {
    const third = option({ id: "mid", base_price: "260", total_journey_minutes: 260 });
    const initial = choosePreferencePair([direct, cheapConnection, third], emptyPreferenceProfile());
    expect(initial).not.toBeNull();
    const profile = { ...emptyPreferenceProfile(), seenPairs: [pairKey(initial!.left, initial!.right)] };
    const next = choosePreferencePair([direct, cheapConnection, third], profile);
    expect(next).not.toBeNull();
    expect(pairKey(next!.left, next!.right)).not.toBe(pairKey(initial!.left, initial!.right));
  });

  it("generalises to held-out pairs from a synthetic preference function", () => {
    const choices = [
      direct,
      cheapConnection,
      option({ id: "early-direct", base_price: "350", total_journey_minutes: 165, departure_at: "2026-12-18T05:00:00Z", arrival_at: "2026-12-18T07:45:00Z" }),
      option({ id: "late-cheap", base_price: "170", total_journey_minutes: 300, departure_at: "2026-12-18T20:00:00Z", arrival_at: "2026-12-18T01:00:00Z", is_nonstop: false, connection_minutes: 70, ticketing_type: "separate_tickets" }),
      option({ id: "protected", base_price: "280", total_journey_minutes: 285, is_nonstop: false, connection_minutes: 105, ticketing_type: "single_ticket" }),
      option({ id: "slow-budget", base_price: "95", total_journey_minutes: 510, is_nonstop: false, connection_minutes: 240, ticketing_type: "separate_tickets", baggage_confidence: "low" }),
      option({ id: "balanced", base_price: "230", total_journey_minutes: 230, departure_at: "2026-12-18T10:00:00Z", arrival_at: "2026-12-18T13:50:00Z" }),
    ];
    const truth: PreferenceWeights = {
      affordability: 2.4,
      journey_efficiency: 1.1,
      nonstop: 0.35,
      departure_convenience: 0.8,
      arrival_convenience: 0.3,
      transfer_comfort: 0.1,
      ticket_protection: 0.55,
      baggage_confidence: 0.2,
    };
    const truthScore = (candidate: typeof direct) => preferenceScore(candidate, truth);
    let profile = emptyPreferenceProfile();
    const trainedPairs = new Set<string>();
    for (let index = 0; index < 12; index += 1) {
      const pair = choosePreferencePair(choices, profile);
      if (!pair) break;
      trainedPairs.add(pairKey(pair.left, pair.right));
      const chosen = truthScore(pair.left) >= truthScore(pair.right) ? pair.left : pair.right;
      const rejected = chosen.id === pair.left.id ? pair.right : pair.left;
      profile = learnPreference(profile, chosen, rejected);
    }

    let correct = 0;
    let total = 0;
    for (let left = 0; left < choices.length; left += 1) {
      for (let right = left + 1; right < choices.length; right += 1) {
        if (trainedPairs.has(pairKey(choices[left], choices[right]))) continue;
        const trueOrder = Math.sign(truthScore(choices[left]) - truthScore(choices[right]));
        const learnedOrder = Math.sign(preferenceScore(choices[left], profile.weights) - preferenceScore(choices[right], profile.weights));
        if (trueOrder === learnedOrder) correct += 1;
        total += 1;
      }
    }
    expect(profile.comparisons).toBe(12);
    expect(total).toBeGreaterThanOrEqual(8);
    expect(correct / total).toBeGreaterThanOrEqual(0.75);
  });

  it("rejects malformed or incompatible saved models", () => {
    expect(parsePreferenceProfile("not json")).toEqual(emptyPreferenceProfile());
    expect(parsePreferenceProfile(JSON.stringify({ version: 99, comparisons: 4, weights: {} }))).toEqual(emptyPreferenceProfile());
  });
});
