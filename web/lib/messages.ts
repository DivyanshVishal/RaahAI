import { z } from "zod";

export const date = z.iso.date();
const timestamp = z.string().regex(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+05:30$/)
  .refine((value) => Number.isFinite(Date.parse(value)), "Invalid timestamp");
const number = z.number();
const nullable = number.nullable();
const stopIndex = z.int().nonnegative();
const downstream = z.int().positive();
const station = z.string();
const scheduled = timestamp.nullable();
const coordinates = { lat: nullable, lng: nullable };
const trainFields = {
  train_no: z.string(),
  train_name: z.string().nullable(),
  train_type: z.string(),
  journey_date: date,
  journey_id: z.int().positive(),
};
const train = z.strictObject(trainFields);
const stop = z.strictObject({
  stop_no: stopIndex,
  station,
  ...coordinates,
  scheduled_arrival: scheduled,
  scheduled_departure: scheduled,
});
const journey = z.strictObject({ ...trainFields, stops: z.array(stop).min(2) });
const forecastFields = {
  predicted_delay_min: nullable,
  range_low_min: nullable,
  range_high_min: nullable,
};
const prediction = z.strictObject({
  stop_no: downstream,
  station,
  ...coordinates,
  scheduled_arrival: scheduled,
  stops_ahead: downstream,
  ...forecastFields,
  expected_arrival: scheduled,
  live_model_delay_min: nullable,
  pre_departure_delay_min: nullable,
});
const completed = z.union([
  z.strictObject({
    stop_no: z.literal(0),
    station,
    scheduled_departure: timestamp,
    actual_departure: timestamp,
    delay_min: number,
  }),
  z.strictObject({
    stop_no: downstream,
    station,
    scheduled_arrival: timestamp,
    actual_arrival: timestamp,
    delay_min: number,
  }),
]);
const actualTrigger = { station, scheduled: timestamp, actual: timestamp, delay_min: number };
const preTrigger = z.strictObject({ event: z.literal("pre_departure"), station, scheduled });
const eventTrigger = z.discriminatedUnion("event", [
  z.strictObject({ event: z.literal("arrival"), stop_no: downstream, ...actualTrigger }),
  z.strictObject({ event: z.literal("departure"), stop_no: stopIndex, ...actualTrigger }),
]);
const forecastMode = z.enum(["pre_departure", "live"]);
const check = z.strictObject({
  station,
  stop_no: downstream,
  ...forecastFields,
  actual_delay_min: number,
  error_min: nullable,
  was_in_range: z.boolean(),
  predicted_at: timestamp,
  prediction_mode: forecastMode,
});
const updateFields = { type: z.literal("prediction_update"), sim_time: timestamp, train };
const update = z.discriminatedUnion("mode", [
  z.strictObject({
    ...updateFields,
    mode: z.literal("pre_departure"),
    trigger: preTrigger,
    current_delay_min: z.null(),
    predictions: z.array(prediction.extend({ live_model_delay_min: z.null() })).min(1),
    completed_stops: z.tuple([]),
    check: z.null(),
  }),
  z.strictObject({
    ...updateFields,
    mode: z.literal("live"),
    trigger: eventTrigger,
    current_delay_min: number,
    predictions: z.array(prediction).min(1),
    completed_stops: z.array(completed),
    check: check.nullable(),
  }),
  z.strictObject({
    ...updateFields,
    mode: z.enum(["finished", "no_live_data"]),
    trigger: eventTrigger,
    current_delay_min: number,
    predictions: z.tuple([]),
    completed_stops: z.array(completed),
    check: check.nullable(),
  }),
]);
const stats = z.strictObject({
  updates_sent: z.int().nonnegative(),
  checks: z.int().nonnegative(),
  mean_abs_error_min: number.nonnegative().nullable(),
  within_range_pct: number.min(0).max(100).nullable(),
});
const settings = z.strictObject({
  date,
  speed: number,
  flush_seconds: number,
  sim_start: timestamp,
  sim_end: timestamp,
  note: z.literal("Replay of real journeys from the test month; predictions use only information available at each simulated moment."),
});

export const messageSchema = z.discriminatedUnion("type", [
  z.strictObject({
    type: z.literal("simulation_start"),
    sent_at: timestamp,
    settings,
    journeys: z.array(journey).min(1),
  }),
  z.strictObject({
    type: z.literal("batch"),
    batch_no: z.int().positive(),
    sent_at: timestamp,
    sim_time: timestamp,
    updates: z.array(update).min(1),
    stats,
  }),
  z.strictObject({ type: z.literal("simulation_end"), sent_at: timestamp, sim_time: timestamp, stats }),
]);

export type Message = z.infer<typeof messageSchema>;
export type Journey = z.infer<typeof journey>;
export type Prediction = z.infer<typeof prediction>;
export type Update = z.infer<typeof update>;
export type Check = z.infer<typeof check>;
export type Stats = z.infer<typeof stats>;
