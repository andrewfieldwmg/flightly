"use client";

import { useCallback, useEffect, useState } from "react";
import type { TripOption } from "@/lib/api/types";
import {
  emptyPreferenceProfile,
  learnPreference,
  parsePreferenceProfile,
  PREFERENCE_STORAGE_KEY,
  type PreferenceProfile,
} from "@/lib/search/preference-model";
import release from "@/lib/search/model-release.json";

export function initialRegisteredProfile(saved: string | null): PreferenceProfile {
  const stored = parsePreferenceProfile(saved);
  if (stored.comparisons > 0) return stored;
  return { ...emptyPreferenceProfile(), weights: { ...release.weights } };
}

export function usePreferenceModel() {
  const [profile, setProfile] = useState<PreferenceProfile>(() => (
    typeof window === "undefined"
      ? initialRegisteredProfile(null)
      : initialRegisteredProfile(window.localStorage.getItem(PREFERENCE_STORAGE_KEY))
  ));

  useEffect(() => {
    window.localStorage.setItem(PREFERENCE_STORAGE_KEY, JSON.stringify(profile));
  }, [profile]);

  const choose = useCallback((chosen: TripOption, rejected: TripOption) => {
    setProfile((current) => learnPreference(current, chosen, rejected));
  }, []);

  const reset = useCallback(() => setProfile(initialRegisteredProfile(null)), []);

  return { profile, choose, reset };
}
