import type { Message } from "./messages";
import { advance, initialState, type ReplayState } from "./replay-state";

type Listener = (message: Message) => void;
const runtime = globalThis as typeof globalThis & {
  railReplayStore?: { state: ReplayState; listeners: Set<Listener> };
};
const store = runtime.railReplayStore ??= { state: initialState, listeners: new Set() };

export function getState() {
  return store.state;
}

export function ingest(message: Message) {
  const next = advance(store.state, message);
  if (next === store.state) return;
  store.state = next;
  for (const listener of store.listeners) listener(message);
}

export function subscribe(listener: Listener) {
  store.listeners.add(listener);
  return () => { store.listeners.delete(listener); };
}
