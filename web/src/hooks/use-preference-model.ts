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

export function usePreferenceModel() {
  const [profile, setProfile] = useState<PreferenceProfile>(() => (
    typeof window === "undefined"
      ? emptyPreferenceProfile()
      : parsePreferenceProfile(window.localStorage.getItem(PREFERENCE_STORAGE_KEY))
  ));

  useEffect(() => {
    window.localStorage.setItem(PREFERENCE_STORAGE_KEY, JSON.stringify(profile));
  }, [profile]);

  const choose = useCallback((chosen: TripOption, rejected: TripOption) => {
    setProfile((current) => learnPreference(current, chosen, rejected));
  }, []);

  const reset = useCallback(() => setProfile(emptyPreferenceProfile()), []);

  return { profile, choose, reset };
}
