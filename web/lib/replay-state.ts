import type { Check, Message, Prediction, Stats, Update } from "./messages";

type Start = Extract<Message, { type: "simulation_start" }>;
export type Forecast = Prediction & { sim_time: string; mode: "pre_departure" | "live" };
export type Progress = {
  latest: Update;
  history: Record<number, Forecast[]>;
};
type RecentCheck = { train: Update["train"]; check: Check };

export type ReplayState = {
  start: Start | null;
  phase: "idle" | "running" | "ended";
  time: string | null;
  batchNo: number;
  stats: Stats | null;
  progress: Record<number, Progress>;
  recentChecks: RecentCheck[];
  checkErrors: number[];
};

export const initialState: ReplayState = {
  start: null,
  phase: "idle",
  time: null,
  batchNo: 0,
  stats: null,
  progress: {},
  recentChecks: [],
  checkErrors: [],
};

export function checkAccuracy({ stats, checkErrors }: ReplayState) {
  // A dropped batch can leave the reported total ahead of the checks we received.
  if (!stats?.checks || checkErrors?.length !== stats.checks) return null;

  const absoluteErrors = checkErrors.map(Math.abs).sort((a, b) => a - b);
  const middle = Math.floor(absoluteErrors.length / 2);
  return {
    medianAbsoluteError: (absoluteErrors[Math.ceil(absoluteErrors.length / 2) - 1] + absoluteErrors[middle]) / 2,
    bias: checkErrors.reduce((sum, error) => sum + error, 0) / checkErrors.length,
  };
}

export function advance(state: ReplayState, message: Message): ReplayState {
  if (message.type === "simulation_start") {
    if (state.phase === "running" && state.batchNo === 0 && state.start?.sent_at === message.sent_at &&
      JSON.stringify(state.start) === JSON.stringify(message)) return state;
    return { ...initialState, start: message, phase: "running", time: message.settings.sim_start };
  }
  if (state.phase !== "running") return state;
  if (message.type === "simulation_end") {
    return { ...state, phase: "ended", stats: message.stats, time: message.sim_time };
  }
  if (message.batch_no <= state.batchNo) return state;

  const progress = { ...state.progress };
  const recentChecks = [...state.recentChecks];
  const checkErrors = [...(state.checkErrors ?? [])];
  for (const update of message.updates) {
    const { journey_id } = update.train;
    const history = { ...progress[journey_id]?.history };
    if (update.mode === "live" || update.mode === "pre_departure") {
      for (const row of update.predictions) {
        history[row.stop_no] = [...(history[row.stop_no] ?? []), {
          ...row,
          sim_time: update.sim_time,
          mode: update.mode,
        }];
      }
    }
    progress[journey_id] = { latest: update, history };
    // Warm-up snapshots use the window start as sim_time, while their actual arrival is earlier.
    if (update.check && update.trigger.event === "arrival" &&
      state.start && update.trigger.actual >= state.start.settings.sim_start) {
      recentChecks.push({ train: update.train, check: update.check });
      if (update.check.error_min !== null) checkErrors.push(update.check.error_min);
    }
  }
  return {
    ...state, progress, recentChecks: recentChecks.slice(-8), checkErrors,
    batchNo: message.batch_no, time: message.sim_time, stats: message.stats,
  };
}
