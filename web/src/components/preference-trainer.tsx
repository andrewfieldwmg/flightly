"use client";

import type { TripOption } from "@/lib/api/types";
import {
  choosePreferencePair,
  leadingPreferences,
  type PreferenceProfile,
} from "@/lib/search/preference-model";
import { money } from "./price-display";
import { duration } from "./result-card";

function clock(value: string): string {
  return value.match(/T(\d{2}:\d{2})/)?.[1] ?? "—";
}

function Choice({ option, onChoose }: { option: TripOption; onChoose: () => void }) {
  return <button type="button" className="preference-choice" onClick={onChoose}>
    <span className="preference-choice-route">{option.route.join(" → ")}</span>
    <strong>{money(option.base_price, option.currency)}</strong>
    <span>{clock(option.departure_at)}–{clock(option.arrival_at)} · {duration(option.total_journey_minutes)}</span>
    <span>{option.is_nonstop ? "Direct" : `${option.connection_airport} transfer · ${duration(option.connection_minutes ?? 0)}`}</span>
  </button>;
}

export function PreferenceTrainer({
  options,
  profile,
  onChoose,
  onReset,
}: {
  options: TripOption[];
  profile: PreferenceProfile;
  onChoose: (chosen: TripOption, rejected: TripOption) => void;
  onReset: () => void;
}) {
  const pair = choosePreferencePair(options, profile);
  const priorities = leadingPreferences(profile);

  if (!pair && !profile.comparisons) return null;

  return <aside className="preference-trainer" aria-label="Personal preference trainer">
    <div className="preference-copy">
      <span className="preference-eyebrow">Preference engine · on-device ML</span>
      <strong>{pair ? "Which flight would you choose?" : "Your current preference model"}</strong>
      <span>{profile.comparisons
        ? `Learned from ${profile.comparisons} choice${profile.comparisons === 1 ? "" : "s"}. ${priorities.length ? `Strongest signals: ${priorities.map((item) => `${item.weight >= 0 ? "more " : "less "}${item.label}`).join(", ")}.` : "Keep training to reveal stronger signals."}`
        : "Teach Flightly your trade-offs. Each answer updates a pairwise ranking model."}</span>
      <small>Stored only in this browser. Feasibility and connection-safety rules always remain hard constraints.</small>
    </div>
    {pair && <div className="preference-choices">
      <Choice option={pair.left} onChoose={() => onChoose(pair.left, pair.right)} />
      <span className="preference-or">or</span>
      <Choice option={pair.right} onChoose={() => onChoose(pair.right, pair.left)} />
    </div>}
    {!!profile.comparisons && <button type="button" className="preference-reset" onClick={onReset}>Reset learning</button>}
  </aside>;
}
