import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { option, synthetic } from "@/test/fixtures";
import { useSharedFeedback } from "./use-shared-feedback";

const KEY = "flightly.shared-feedback.v1";
beforeEach(() => localStorage.clear());
afterEach(() => vi.unstubAllGlobals());

describe("consented shared feedback", () => {
  it("keeps choices local until opt-in and sends only verified option references", async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", fetcher);
    const { result } = renderHook(() => useSharedFeedback("stored-search"));
    act(() => result.current.share(option({ id: "a" }), synthetic()));
    expect(fetcher).not.toHaveBeenCalled();
    act(() => result.current.toggle(true));
    act(() => result.current.share(option({ id: "a" }), synthetic()));
    await waitFor(() => expect(result.current.status).toContain("Choice shared"));
    const payload = JSON.parse(fetcher.mock.calls[0][1].body);
    expect(payload).toMatchObject({ consent: true, search_id: "stored-search", chosen_id: "a", rejected_id: "synthetic-OUTBOUND" });
    expect(payload).not.toHaveProperty("base_price");
    expect(payload).not.toHaveProperty("route");
    expect(JSON.parse(localStorage.getItem(KEY)!).pending).toHaveLength(0);
  });

  it("retries the same event ID after a network failure and clears pending work on opt-out", async () => {
    const fetcher = vi.fn().mockRejectedValueOnce(new Error("offline")).mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", fetcher);
    const { result } = renderHook(() => useSharedFeedback("stored-search"));
    act(() => result.current.toggle(true));
    act(() => result.current.share(option({ id: "a" }), synthetic()));
    await waitFor(() => expect(result.current.status).toContain("Sharing pending"));
    const eventId = JSON.parse(fetcher.mock.calls[0][1].body).event_id;
    act(() => window.dispatchEvent(new Event("online")));
    await waitFor(() => expect(result.current.status).toContain("Choice shared"));
    expect(JSON.parse(fetcher.mock.calls[1][1].body).event_id).toBe(eventId);
    act(() => result.current.toggle(false));
    expect(JSON.parse(localStorage.getItem(KEY)!)).toMatchObject({ enabled: false, pending: [] });
  });
});
