import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { option, synthetic } from "@/test/fixtures";
import { emptyPreferenceProfile } from "@/lib/search/preference-model";
import { PreferenceTrainer } from "./preference-trainer";
import { DirectionResults } from "./direction-results";

const direct = option({ id: "direct" });
const connection = synthetic();

describe("preference trainer", () => {
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
