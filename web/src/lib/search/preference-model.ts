import type { TripOption } from "@/lib/api/types";

export const PREFERENCE_MODEL_VERSION = 1 as const;
export const PREFERENCE_STORAGE_KEY = "flightly.preference-model.v1";

export const preferenceFeatureKeys = [
  "affordability",
  "journey_efficiency",
  "nonstop",
  "departure_convenience",
  "arrival_convenience",
  "transfer_comfort",
  "ticket_protection",
  "baggage_confidence",
] as const;

export type PreferenceFeatureKey = (typeof preferenceFeatureKeys)[number];
export type PreferenceWeights = Record<PreferenceFeatureKey, number>;

export interface PreferenceProfile {
  version: typeof PREFERENCE_MODEL_VERSION;
  weights: PreferenceWeights;
  comparisons: number;
  seenPairs: string[];
  updatedAt: string | null;
}

export interface PreferenceTrainingPair {
  left: TripOption;
  right: TripOption;
  uncertainty: number;
}

export const preferenceFeatureLabels: Record<PreferenceFeatureKey, string> = {
  affordability: "lower prices",
  journey_efficiency: "shorter journeys",
  nonstop: "direct flights",
  departure_convenience: "convenient departures",
  arrival_convenience: "earlier arrivals",
  transfer_comfort: "comfortable transfers",
  ticket_protection: "protected tickets",
  baggage_confidence: "clear baggage pricing",
};

const zeroWeights = (): PreferenceWeights => Object.fromEntries(
  preferenceFeatureKeys.map((key) => [key, 0]),
) as PreferenceWeights;

export function emptyPreferenceProfile(): PreferenceProfile {
  return {
    version: PREFERENCE_MODEL_VERSION,
    weights: zeroWeights(),
    comparisons: 0,
    seenPairs: [],
    updatedAt: null,
  };
}

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function clockMinutes(value: string): number {
  const match = value.match(/T(\d{2}):(\d{2})/);
  return match ? Number(match[1]) * 60 + Number(match[2]) : 12 * 60;
}

function confidenceValue(value: string): number {
  const normalized = value.toLowerCase();
  if (normalized === "high" || normalized === "complete") return 1;
  if (normalized === "medium" || normalized === "estimated") return 0.6;
  if (normalized === "low" || normalized === "partial") return 0.25;
  return 0;
}

/**
 * Explainable, bounded features. Every feature is oriented so a larger value
 * means "more of the named benefit"; the learned weight supplies the user's
 * preference, including aversion when it is negative.
 */
export function preferenceFeatures(option: TripOption): PreferenceWeights {
  const price = Math.max(0, Number(option.base_price) || 0);
  const departure = clockMinutes(option.departure_at);
  const arrival = clockMinutes(option.arrival_at);
  const transferMinutes = option.connection_minutes ?? 0;

  return {
    affordability: 1 / (1 + price / 250),
    journey_efficiency: 1 / (1 + Math.max(0, option.total_journey_minutes) / 240),
    nonstop: option.is_nonstop ? 1 : 0,
    departure_convenience: clamp(1 - Math.abs(departure - 10 * 60) / (10 * 60)),
    arrival_convenience: clamp(1 - Math.max(0, arrival - 18 * 60) / (6 * 60)),
    transfer_comfort: option.is_nonstop ? 0 : clamp((transferMinutes - 45) / 180),
    ticket_protection: option.is_nonstop || option.ticketing_type === "single_ticket" ? 1 : 0,
    baggage_confidence: Math.max(confidenceValue(option.baggage_confidence), confidenceValue(option.price_completeness)),
  };
}

export function preferenceScore(option: TripOption, weights: PreferenceWeights): number {
  const features = preferenceFeatures(option);
  return preferenceFeatureKeys.reduce((score, key) => score + features[key] * weights[key], 0);
}

function sigmoid(value: number): number {
  if (value >= 0) return 1 / (1 + Math.exp(-Math.min(value, 40)));
  const exponential = Math.exp(Math.max(value, -40));
  return exponential / (1 + exponential);
}

export function pairKey(first: TripOption, second: TripOption): string {
  return [first.id, second.id].sort().join("::");
}

/** One online pairwise-logistic update, with light L2 regularisation. */
export function learnPreference(
  profile: PreferenceProfile,
  chosen: TripOption,
  rejected: TripOption,
  learningRate = 0.8,
): PreferenceProfile {
  const chosenFeatures = preferenceFeatures(chosen);
  const rejectedFeatures = preferenceFeatures(rejected);
  const weights = { ...profile.weights };
  const regularisation = 0.01;

  // A few small SGD passes make a single explicit choice visible immediately
  // without pretending that one comparison creates a settled preference.
  for (let epoch = 0; epoch < 4; epoch += 1) {
    const margin = preferenceFeatureKeys.reduce(
      (sum, key) => sum + weights[key] * (chosenFeatures[key] - rejectedFeatures[key]),
      0,
    );
    const residual = 1 - sigmoid(margin);
    for (const key of preferenceFeatureKeys) {
      const delta = chosenFeatures[key] - rejectedFeatures[key];
      weights[key] += learningRate * (residual * delta - regularisation * weights[key]);
    }
  }

  const seenPairs = [...profile.seenPairs.filter((key) => key !== pairKey(chosen, rejected)), pairKey(chosen, rejected)].slice(-120);
  return {
    version: PREFERENCE_MODEL_VERSION,
    weights,
    comparisons: profile.comparisons + 1,
    seenPairs,
    updatedAt: new Date().toISOString(),
  };
}

/**
 * Active learning: choose the unseen pair closest to a 50/50 prediction.
 * Ties favor pairs with a larger feature distance because they are more
 * informative while the model is still cold.
 */
export function choosePreferencePair(options: TripOption[], profile: PreferenceProfile): PreferenceTrainingPair | null {
  const candidates = options.slice(0, 18);
  let best: (PreferenceTrainingPair & { distance: number }) | null = null;
  const seen = new Set(profile.seenPairs);

  for (let leftIndex = 0; leftIndex < candidates.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < candidates.length; rightIndex += 1) {
      const left = candidates[leftIndex];
      const right = candidates[rightIndex];
      if (seen.has(pairKey(left, right))) continue;
      const leftFeatures = preferenceFeatures(left);
      const rightFeatures = preferenceFeatures(right);
      const distance = preferenceFeatureKeys.reduce(
        (sum, key) => sum + Math.abs(leftFeatures[key] - rightFeatures[key]),
        0,
      );
      if (distance < 0.08) continue;
      const probability = sigmoid(preferenceScore(left, profile.weights) - preferenceScore(right, profile.weights));
      const uncertainty = Math.abs(probability - 0.5);
      if (!best || uncertainty < best.uncertainty - 1e-9 || (Math.abs(uncertainty - best.uncertainty) < 1e-9 && distance > best.distance)) {
        best = { left, right, uncertainty, distance };
      }
    }
  }

  return best ? { left: best.left, right: best.right, uncertainty: best.uncertainty } : null;
}

export function leadingPreferences(profile: PreferenceProfile, limit = 3): Array<{ key: PreferenceFeatureKey; label: string; weight: number }> {
  if (!profile.comparisons) return [];
  return preferenceFeatureKeys
    .map((key) => ({ key, label: preferenceFeatureLabels[key], weight: profile.weights[key] }))
    .filter((item) => Math.abs(item.weight) > 0.02)
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
    .slice(0, limit);
}

export function parsePreferenceProfile(value: string | null): PreferenceProfile {
  if (!value) return emptyPreferenceProfile();
  try {
    const parsed = JSON.parse(value) as Partial<PreferenceProfile>;
    if (parsed.version !== PREFERENCE_MODEL_VERSION || typeof parsed.comparisons !== "number" || !parsed.weights) return emptyPreferenceProfile();
    const weights = zeroWeights();
    for (const key of preferenceFeatureKeys) {
      const weight = (parsed.weights as Partial<PreferenceWeights>)[key];
      if (typeof weight !== "number" || !Number.isFinite(weight)) return emptyPreferenceProfile();
      weights[key] = weight;
    }
    return {
      version: PREFERENCE_MODEL_VERSION,
      weights,
      comparisons: Math.max(0, Math.floor(parsed.comparisons)),
      seenPairs: Array.isArray(parsed.seenPairs) ? parsed.seenPairs.filter((key): key is string => typeof key === "string").slice(-120) : [],
      updatedAt: typeof parsed.updatedAt === "string" ? parsed.updatedAt : null,
    };
  } catch {
    return emptyPreferenceProfile();
  }
}
