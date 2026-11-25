export const DEFAULT_REPLAY_DATE = "2026-08-20";
export const FIRST_REPLAY_DATE = "2026-08-01";
export const LAST_REPLAY_DATE = "2026-08-31";

export function isReplayDate(value: string) {
  return value >= FIRST_REPLAY_DATE && value <= LAST_REPLAY_DATE;
}
