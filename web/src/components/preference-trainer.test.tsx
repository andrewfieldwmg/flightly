import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { option, synthetic } from "@/test/fixtures";
import { emptyPreferenceProfile } from "@/lib/search/preference-model";
import { PreferenceTrainer } from "./preference-trainer";
import { DirectionResults } from "./direction-results";
import { initialRegisteredProfile } from "@/hooks/use-preference-model";
import release from "@/lib/search/model-release.json";

const direct = option({ id: "direct" });
const connection = synthetic();

describe("preference trainer", () => {
  it("starts from the registry prior and preserves previously trained personal profiles", () => {
    expect(initialRegisteredProfile(null).weights).toEqual(release.weights);
    const saved = { ...emptyPreferenceProfile(), comparisons: 3 };
    saved.weights.affordability = 4;
    expect(initialRegisteredProfile(JSON.stringify(saved))).toEqual(saved);
    expect(initialRegisteredProfile("bad json").weights).toEqual(release.weights);
  });

  it("reranks feasible results using personal learning only when For you is selected", () => {
    const results = {
      baseline: direct,
      nonstop_options: [direct],
      cheapest_feasible: connection,
      fastest_feasible: connection,
      pareto_frontier: [connection],
      feasible_options: [connection],
    };
    const profile = emptyPreferenceProfile();
    profile.weights.affordability = 2;
    const props = {
      title: "Outbound", date: "2026-12-18", results,
      selectedId: null, onSelect: vi.fn(), complete: true,
      connectionProfile: "CONSERVATIVE" as const, selfTransferEnabled: true,
    };
    const { rerender } = render(<DirectionResults {...props} preferenceProfile={profile} />);
    expect(screen.getAllByRole("radio")[0]).toHaveAccessibleName("Select LGW-CAG");
    fireEvent.click(screen.getByRole("button", { name: "For you" }));
    expect(screen.getAllByRole("radio")[0]).toHaveAccessibleName("Select LGW-MXP-CAG");
    const updated = { ...profile, weights: { ...profile.weights, nonstop: 5 } };
    rerender(<DirectionResults {...props} preferenceProfile={updated} />);
    expect(screen.getAllByRole("radio")[0]).toHaveAccessibleName("Select LGW-CAG");
    rerender(<DirectionResults {...props} preferenceProfile={profile} selfTransferEnabled={false} />);
    expect(screen.getAllByRole("radio")).toHaveLength(1);
  });

  it("turns an explicit comparison into a labelled training example", () => {
    const onChoose = vi.fn();
    render(<PreferenceTrainer options={[direct, connection]} profile={emptyPreferenceProfile()} onChoose={onChoose} onReset={vi.fn()} />);

    expect(screen.getByText("Which flight would you choose?")).toBeInTheDocument();
    expect(screen.getByText(/Each answer updates a pairwise ranking model/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /LGW → MXP → CAG/ }));
    expect(onChoose).toHaveBeenCalledWith(connection, direct);
  });

  it("keeps preference training outside an individual direction table", () => {
    const results = {
      baseline: direct,
      nonstop_options: [direct],
      cheapest_feasible: connection,
      fastest_feasible: connection,
      pareto_frontier: [connection],
      feasible_options: [connection],
    };
    render(<DirectionResults
      title="Outbound"
      date="2026-12-18"
      results={results}
      selectedId={null}
      onSelect={vi.fn()}
      complete
      connectionProfile="CONSERVATIVE"
      selfTransferEnabled
    />);

    expect(screen.queryByRole("button", { name: /For you/i })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Personal preference trainer")).not.toBeInTheDocument();
  });
});
