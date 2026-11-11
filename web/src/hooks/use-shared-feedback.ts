"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { resolveApiUrl } from "@/lib/api/client";
import type { TripOption } from "@/lib/api/types";
import release from "@/lib/search/model-release.json";

const KEY = "flightly.shared-feedback.v1";
type Choice = { event_id: string; participant_id: string; consent: true; search_id: string; chosen_id: string; rejected_id: string; model_name: string; model_version: string; profile_version: 1 };
type Stored = { enabled: boolean; participant: string; pending: Choice[] };

function read(): Stored {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) ?? "null");
    if (saved && typeof saved.enabled === "boolean" && typeof saved.participant === "string" && Array.isArray(saved.pending)) {
      const pending = saved.pending.filter((item: Partial<Choice> | null) => item && item.consent === true && typeof item.event_id === "string" && typeof item.search_id === "string" && typeof item.chosen_id === "string" && typeof item.rejected_id === "string").slice(-200);
      return { enabled: saved.enabled, participant: saved.participant, pending: saved.enabled ? pending : [] };
    }
  } catch { /* Storage can be unavailable in private browsing. */ }
  return { enabled: false, participant: crypto.randomUUID(), pending: [] };
}

export function useSharedFeedback(searchId?: string) {
  const [enabled, setEnabled] = useState(false);
  const [status, setStatus] = useState("Sharing is off. Choices stay in this browser.");
  const busy = useRef(false);
  const controller = useRef<AbortController | null>(null);
  const state = useRef<Stored | null>(null);
  const persist = useCallback(() => {
    try { localStorage.setItem(KEY, JSON.stringify(state.current)); } catch {
      setStatus("Browser storage unavailable. Pending feedback cannot survive a reload.");
    }
  }, []);
  const flush = useCallback(async () => {
    if (busy.current || !state.current?.enabled) return;
    busy.current = true;
    try {
      while (state.current.enabled && state.current.pending.length) {
        const choice = state.current.pending[0];
        controller.current = new AbortController();
        const timeout = window.setTimeout(() => controller.current?.abort(), 10_000);
        let response: Response;
        try {
          response = await fetch(resolveApiUrl("/api/learning/feedback"), {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(choice), signal: controller.current.signal,
          });
        } finally { window.clearTimeout(timeout); }
        if (!state.current.enabled) break;
        if (!response.ok && (response.status >= 500 || response.status === 429)) throw new Error("retry");
        state.current.pending = state.current.pending.filter((item) => item.event_id !== choice.event_id);
        persist();
        setStatus(response.ok ? "Choice shared. Personal learning updates immediately; shared models are evaluated before release." : "This choice could not be shared. Your personal model still learned it.");
      }
    } catch {
      if (state.current?.enabled) setStatus("Sharing pending. We will retry in 30 seconds or when you reconnect.");
    } finally { busy.current = false; }
  }, [persist]);
  useEffect(() => {
    state.current = read();
    setEnabled(state.current.enabled);
    if (state.current.enabled) setStatus("Sharing enabled. No routes, travel dates or names are shared.");
    void flush();
    const retry = () => { void flush(); };
    window.addEventListener("online", retry);
    const interval = window.setInterval(retry, 30_000);
    return () => { window.removeEventListener("online", retry); window.clearInterval(interval); controller.current?.abort(); };
  }, [flush]);
  const toggle = useCallback((value: boolean) => {
    if (!state.current) state.current = read();
    state.current.enabled = value;
    if (!value) { state.current.pending = []; controller.current?.abort(); }
    setEnabled(value);
    setStatus(value ? "Sharing enabled. No routes, travel dates or names are shared." : "Sharing is off. Previously shared choices remain in the training dataset.");
    persist();
  }, [persist]);
  const share = useCallback((chosen: TripOption, rejected: TripOption) => {
    if (!state.current?.enabled || !searchId) return;
    if (state.current.pending.length >= 200) { setStatus("Sharing queue full. Your personal model still learned this choice."); return; }
    state.current.pending.push({ event_id: crypto.randomUUID(), participant_id: state.current.participant, consent: true, search_id: searchId, chosen_id: chosen.id, rejected_id: rejected.id, model_name: release.name, model_version: release.version, profile_version: 1 });
    persist();
    void flush();
  }, [flush, persist, searchId]);
  return { enabled, status, toggle, share };
}
