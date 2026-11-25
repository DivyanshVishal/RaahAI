# Train delay pipeline and replay contract

## System and data flow

1. `clean_train_data.py` cleans station-level GPS records. `build_segments.py` turns consecutive stops into A → B rows and adds schedule, recent running state, and train history. `add_network_features.py` adds scheduled traffic and historical network features and chooses the latest known delay at A. The inference inputs are under `processed/features/year=2026`.
2. `train_baseline.py` provides a next-stop baseline. `train_multistop.py` trains six LightGBM models (live and pre-departure, each with p10, p50, and p90). The saved models and `feature_lists.json` define the feature order, category levels, range scales, and blend weights. The default split trains before July 2026, uses July for validation, and holds out August. The bundled training report confirms the August holdout.
3. `simulate_live.py` loads the trained models through `predict.py`, selects journeys by start date, and builds events from scheduled times plus delays from the feature table. It forecasts from the origin 30 simulated minutes before scheduled departure and updates the forecasts at recorded or reconstructed arrival and departure events. It sends a schedule manifest, batches of updates, and final stats. The stream carries actuals as events are replayed. Some underlying arrival delays are imputed during preprocessing, and the wire format carries no imputation flag.
4. `predict.py` runs the same Python models on demand for one journey. `app.py` exposes individual feature inputs for a single A → B what-if, and `pick_journeys.py` selects varied held-out examples for CLI inspection.

## JSON message schema

Source of truth is [`simulate_live.py`](../simulate_live.py), specifically `Journey.stations`, `Journey.train_info`, `Journey.prediction_rows`, `handle`, `stat_block`, `flush`, and the start/end sends. I parsed all 19 JSONL records in [`sim_log.jsonl`](../sim_log.jsonl) and checked the synthetic examples in [`sample_messages.json`](../sample_messages.json). The synthetic file wraps examples in an object with `_note` and uses April dates and placeholder stations. The actual log has one POST body per line and starts on August 20, 2026.

All listed keys are required within their respective shape. `T | null` means a present key with a nullable value. Event-specific keys appear only on the shapes that list them. JSON numbers are finite in emitted messages (`allow_nan=False`); `integer` identifies integral counters and indices. `Date` is a string in `YYYY-MM-DD` format. `Timestamp` is a string in `YYYY-MM-DDTHH:mm:ss+05:30` format, with no fractional seconds. The timestamp helper appends a literal `+05:30`. In particular, `sent_at` formats the Python process's local wall clock and performs no timezone conversion. Schedule and simulation times represent the dataset's Indian Standard Time. Delays are signed minutes, with positive values meaning late. Displayed delays and range bounds are rounded to one decimal and coordinates to five. Example timestamps and values below come from the log unless noted.

### Top-level messages

The POST body is a discriminated union of these three object shapes. A `prediction_update` occurs only within `batch.updates`.

| Message | Field | Type | Meaning | Example |
| --- | --- | --- | --- | --- |
| Start | `type` | literal `"simulation_start"` | Announces the replay and its full schedule manifest | `"simulation_start"` |
| Start | `sent_at` | Timestamp | Creation time on the sender's wall clock | `"2026-10-22T19:26:34+05:30"` |
| Start | `settings` | Settings | Replay parameters | `{"date":"2026-08-20", …}` |
| Start | `journeys` | Journey[] | All selected journeys with all scheduled stops, including journeys outside the replay window | Array of 30 journeys in the log |
| Batch | `type` | literal `"batch"` | Delivers a nonempty group of updates | `"batch"` |
| Batch | `batch_no` | integer ≥ 1 | Sequence number within one replay, incremented at each nonempty flush | `1` |
| Batch | `sent_at` | Timestamp | Creation time on the sender's wall clock | `"2026-10-22T19:26:36+05:30"` |
| Batch | `sim_time` | Timestamp | Simulator clock at flush, shared at batch level | `"2026-08-20T05:36:00+05:30"` |
| Batch | `updates` | PredictionUpdate[] | Update objects in emission order | Array of 50 updates in batch 1 |
| Batch | `stats` | Stats | Cumulative scoring snapshot through the flush | `{"updates_sent":50,"checks":27,"mean_abs_error_min":1.43,"within_range_pct":77.8}` |
| End | `type` | literal `"simulation_end"` | Announces replay termination after the final nonempty flush, including Ctrl+C during the replay loop | `"simulation_end"` |
| End | `sent_at` | Timestamp | Creation time on the sender's wall clock | `"2026-10-22T19:27:20+05:30"` |
| End | `sim_time` | Timestamp | Simulator clock when the loop ended | `"2026-08-22T13:41:00+05:30"` |
| End | `stats` | Stats | Final cumulative scoring snapshot | `{"updates_sent":825,"checks":437,"mean_abs_error_min":3.02,"within_range_pct":80.1}` |

### Settings

| Field | Type | Meaning | Example |
| --- | --- | --- | --- |
| `date` | Date | Selected journey start date | `"2026-08-20"` |
| `speed` | number | Simulated seconds per real second | `60.0` |
| `flush_seconds` | number | Configured interval in real seconds for a normal flush | `2.0` |
| `sim_start` | Timestamp | Start of the requested window, or first event time | `"2026-08-20T02:45:00+05:30"` |
| `sim_end` | Timestamp | End of the requested window, or last event time | `"2026-08-22T13:41:00+05:30"` |
| `note` | string, fixed text | Sender's description of the replay | `"Replay of real journeys from the test month; predictions use only information available at each simulated moment."` |

### TrainInfo and Journey

An update's `train` is TrainInfo. Each start message's Journey has all TrainInfo keys plus `stops`. Journey IDs are assigned from 1 for each replay, and stop numbers are route positions, so the same station code can identify multiple positions.

| Field | Type | Meaning | Example |
| --- | --- | --- | --- |
| `train_no` | string | Train number kept as text | `"12189"` |
| `train_name` | string or null | Name looked up from matching cleaned parquet, or null if unavailable | `"MAHAKAUSHAL EXP"` |
| `train_type` | string | Dataset category converted to text | `"SUF"` |
| `journey_date` | Date | Journey start date, even for stops on later days | `"2026-08-20"` |
| `journey_id` | integer ≥ 1 | Replay-local journey identifier | `1` |
| `stops` (Journey only) | Stop[] | Route order, origin followed by each segment's destination | Array beginning with stop 0 at `"JBP"` |

### Stop

| Field | Type | Meaning | Example |
| --- | --- | --- | --- |
| `stop_no` | integer ≥ 0 | Zero-based route position | `0` |
| `station` | string | Station code | `"JBP"` |
| `lat` | number or null | Latitude | `23.16516` |
| `lng` | number or null | Longitude | `79.95125` |
| `scheduled_arrival` | Timestamp or null | Arrival schedule, null at the origin or when unavailable | `null` |
| `scheduled_departure` | Timestamp or null | Departure schedule, null at the terminal stop or when unavailable | `"2026-08-20T18:10:00+05:30"` |

### PredictionUpdate

| Field | Type | Meaning | Example |
| --- | --- | --- | --- |
| `type` | literal `"prediction_update"` | Update discriminator | `"prediction_update"` |
| `sim_time` | Timestamp | Event time, or the replay-window start for a warm-up snapshot | `"2026-08-20T04:14:00+05:30"` |
| `train` | TrainInfo | Identity and metadata for this journey | `{"train_no":"18636","train_name":"RANCHI EXPRESS","train_type":"MEX","journey_date":"2026-08-20","journey_id":29}` |
| `trigger` | Trigger | Event that caused the forecast or status update | Arrival at `"NBG"` |
| `mode` | `"pre_departure"` or `"live"` or `"finished"` or `"no_live_data"` | Forecast or status branch | `"live"` |
| `current_delay_min` | number or null | Known model-input delay in live mode, event delay in finished or no-live-data mode, null before departure | `12.0` |
| `predictions` | Prediction[] | Ordered downstream arrival forecasts. Finished and no-live-data updates carry an empty array | `[]` |
| `completed_stops` | CompletedStop[] | Cumulative snapshot of known stop records, in event insertion order. Pre-departure carries an empty array | Array containing the arrival at `"NBG"` |
| `check` | Check or null | Comparison with the preceding forecast for the arrived stop, when available | NBG check with `error_min` of `1.9` |

### Trigger

Discriminated by `event`. The pre-departure variant has exactly the first three keys. Both actual-event variants include all six. Actual events are filtered out if they lack a schedule-plus-delay timestamp, so their schedule, actual time, and delay are present as non-null values.

| Field | Type | Meaning | Example |
| --- | --- | --- | --- |
| `event` | `"pre_departure"` or `"arrival"` or `"departure"` | Advance forecast, arrival event, or departure event | `"arrival"` |
| `station` | string | Station code at the event, or origin code for pre-departure | `"NBG"` |
| `scheduled` | Timestamp for actual events, Timestamp or null for pre-departure | Scheduled arrival for an arrival, scheduled departure otherwise. The pre-departure event itself requires a usable origin schedule | `"2026-08-20T04:02:00+05:30"` |
| `stop_no` (actual events only) | integer ≥ 0 | Route position of the triggering stop. Arrival positions start at 1 | `1` |
| `actual` (actual events only) | Timestamp | Schedule plus the event delay | `"2026-08-20T04:14:00+05:30"` |
| `delay_min` (actual events only) | number | The delay for this arrival or departure event | `12.0` |

### Prediction

| Field | Type | Meaning | Example |
| --- | --- | --- | --- |
| `stop_no` | integer ≥ 1 | Destination stop's route position | `1` |
| `station` | string | Destination station code | `"NBG"` |
| `lat` | number or null | Destination latitude | `24.65324` |
| `lng` | number or null | Destination longitude | `84.09387` |
| `scheduled_arrival` | Timestamp or null | Scheduled arrival at destination | `"2026-08-20T04:02:00+05:30"` |
| `stops_ahead` | integer ≥ 1 | One-based position in this forecast's list of downstream stops | `1` |
| `predicted_delay_min` | number or null | Central forecast (BEST for live, PRE before departure) | `13.9` |
| `range_low_min` | number or null | Calibrated lower bound in delay minutes | `0.7` |
| `range_high_min` | number or null | Calibrated upper bound in delay minutes | `92.0` |
| `expected_arrival` | Timestamp or null | Schedule plus the unrounded central forecast, serialized to seconds | `"2026-08-20T04:15:52+05:30"` |
| `live_model_delay_min` | number or null | Live model median before blending. Null on all pre-departure rows | `null` |
| `pre_departure_delay_min` | number or null | Pre-departure model median for the same destination | `13.9` |

### CompletedStop

Two shapes distinguished by their time keys. The origin is included when its departure is replayed. Each subsequent entry represents an arrival. Intermediate departure times appear on departure triggers. The completed-stop array repeats prior entries.

| Variant | Field | Type | Meaning | Example |
| --- | --- | --- | --- | --- |
| Both | `stop_no` | integer | Route position, 0 for origin departure and ≥ 1 for arrivals | `0` |
| Both | `station` | string | Station code | `"LKU"` |
| Both | `delay_min` | number | Observed event delay, using the feature table's value for that event | `0.0` |
| Origin departure | `scheduled_departure` | Timestamp | Origin departure schedule | `"2026-08-20T04:25:00+05:30"` |
| Origin departure | `actual_departure` | Timestamp | Origin departure derived from schedule plus delay | `"2026-08-20T04:25:00+05:30"` |
| Arrival | `scheduled_arrival` | Timestamp | Schedule at the arrived stop | `"2026-08-20T04:02:00+05:30"` |
| Arrival | `actual_arrival` | Timestamp | Arrival derived from schedule plus delay | `"2026-08-20T04:14:00+05:30"` |

### Check

On arrival, the simulator looks up and removes the most recent stored forecast for this destination. If one exists, it compares the arrival delay to the unrounded central forecast and bounds. The signed error is forecast minus actual. A positive error means the forecast was later than the actual. Boundaries are inclusive. Serialized error and bounds are rounded separately, so deriving a check from those displayed values can differ from the simulator's answer. A check can appear on a final arrival with no onward forecasts.

| Field | Type | Meaning | Example |
| --- | --- | --- | --- |
| `station` | string | Station just reached | `"NBG"` |
| `stop_no` | integer ≥ 1 | Position of the arrived stop | `1` |
| `predicted_delay_min` | number or null | Last central forecast for this stop | `13.9` |
| `range_low_min` | number or null | Its lower bound | `0.7` |
| `range_high_min` | number or null | Its upper bound | `92.0` |
| `actual_delay_min` | number | Delay for the arrival event | `12.0` |
| `error_min` | number or null | Signed forecast minus actual, in minutes | `1.9` |
| `was_in_range` | boolean | Whether actual lay within the unrounded bounds | `true` |
| `predicted_at` | Timestamp | Simulated time when that earlier forecast was made | `"2026-08-20T02:45:00+05:30"` |
| `prediction_mode` | `"pre_departure"` or `"live"` | Mode of that earlier forecast | `"pre_departure"` |

### Stats

| Field | Type | Meaning | Example |
| --- | --- | --- | --- |
| `updates_sent` | integer ≥ 0 | Number of updates counted at flush, including dry-run and failed-delivery batches | `825` |
| `checks` | integer ≥ 0 | Number of arrivals scored against a prior forecast in the scored interval | `437` |
| `mean_abs_error_min` | number or null | Cumulative mean absolute error from internal errors, rounded to two decimals. Null before any checks | `3.02` |
| `within_range_pct` | number or null | Cumulative percentage of checked arrivals inside the bounds, rounded to one decimal on a 0–100 scale. Null before any checks | `80.1` |

## Forecast meaning and replay behavior

- **BEST and low/high.** The live model predicts the change from the known delay at A to the arrival delay at B. Python adds the known delay back to each quantile. The pre-departure model predicts arrival delay directly from schedule and history features. Python sorts the three quantiles to avoid crossing. For BEST it blends the two sets of quantiles using the July-fitted live weight for the number of stops ahead and current delay, then uses the central value. Without blend metadata, the live median supplies the live central value. Before departure the central value is the pre-departure median. Low/high are the calibrated p10/p90 distances around the central value, aiming for about 80% coverage across forecasts. In the bundled metadata the scale is 1.12 for live and blend, and 1.08 for pre-departure. The bundled blend uses 100% live weight through 10 stops ahead and mixes in PRE for some longer forecasts.
- **`stops_ahead`.** This is the number of route stops from the forecast's anchor to the target, so the very next stop is 1. In `make_pairs`, it is `bj - ai + 1`, with `ai` the segment leaving A and `bj` the segment arriving at B. Absolute `stop_no` stays fixed when the forecast anchor changes.
- **`mode` and `trigger`.** Pre-departure forecasts cover all downstream arrivals. Both actual-event triggers can produce `live` forecasts. At an arrival, the simulator uses the arrival delay as known at the new anchor and blanks four departure-dependent features there (`dep_delay_A`, `dwell_overrun_A`, `trend_2stops`, `max_dep_delay_so_far`). At a departure it uses the feature row's known delay, which prefers departure. `finished` requires an arrival at the final segment's destination. `no_live_data` means an onward live prediction returned no result because the current input delay was unavailable. The trigger still records the event and the completed stops still carry known actuals. The log contains pre-departure, live, and finished updates. The code also defines the no-live-data branch.
- **`check` and history.** A check scores the latest stored forecast for that stop, including PRE if it was the last forecast available. Only arrivals can produce checks. Forecast history is implicit across updates, so a chart needs to retain rows per replay, journey, and stop with their update times. `completed_stops` is a snapshot, so counting every entry on every update would repeat arrivals.
- **Window warm-up.** With `--start-time`, earlier events are processed internally with the window start as their update timestamp. The replay retains the last warm-up snapshot per journey unless its mode is finished, and then resets the accuracy counters before scoring the window. A retained snapshot can carry a pre-window check even though that check is excluded from the stats. Its actual trigger timestamp can precede the update's `sim_time`. Regular event updates use event time. The announced end is a window boundary or last event time, and the end message reports the loop clock at termination.

## CLI and transport

| Option | Default | Behavior |
| --- | --- | --- |
| `--url` | `http://localhost:3000/api/predictions` | Receives JSON POSTs with `Content-Type: application/json` |
| `--date` | `2026-08-20` | Selects journeys starting on that date. The CLI can accept other dates with available data |
| `--trains` | `30` | Samples up to that many journeys from the eligible pool |
| `--train-list` | none | Selects matching train numbers directly. Bypasses random sampling, stop minimum and 80% arrival-coverage filter |
| `--min-stops`, `--seed` | `6`, `7` | Minimum route length and random seed for sampling. Random candidates also need at least 80% arrival-delay coverage |
| `--speed` | `60.0` | Simulated seconds per real second |
| `--flush` | `2.0` | Seconds between normal flush attempts. Empty buffers produce no batch |
| `--start-time` | none | Window start in `HH:MM` on the selected date |
| `--hours` | none | Sets window end when a start time and a nonzero duration are supplied |
| `--dry-run` | off | Generates and logs all messages without making HTTP requests |
| `--fast` | off | Skips waiting and flushes once the buffer reaches at least 50 updates, plus a final flush |
| `--features`, `--clean`, `--models` | `processed/features/year=2026`, `processed/clean_data/year=2026`, `models/multistop` | Input roots relative to the process working directory. Clean data is used to look up train names |
| `--log`, `--outbox`, `--threads` | `sim_log.jsonl`, `sim_outbox.jsonl`, `8` | Output paths and worker count for journey discovery |

The sender opens the log in overwrite mode and writes and flushes each exact JSON body before trying delivery. In normal mode there are up to three synchronous attempts per message with a 5-second request timeout. A 2xx status is success. Caught network, OS, and value errors cause backoffs of 0.3, 0.6, and 0.9 seconds, including after the final caught failure. If delivery fails, the sender appends that body to the outbox and moves on. There is no automatic resend from the outbox. An accepted POST whose response is lost can be delivered again. The wire has no run identifier and the batch sequence restarts at 1. A state store can use the start message as the run boundary and the batch number to recognize a repeated batch within that run.

For the Phase 2 smoke test, dry-run generation and HTTP replay are separate steps. Use a separate `--log` path to preserve the existing fixture, or POST the existing `sim_log.jsonl` in order to test the real recorded contract. The existing fixture contains 30 journeys, 467 scheduled stops, 825 updates, and 437 checks. Its final snapshot is 3.02 min MAE and 80.1% inside the range.

## Manual prediction contract

- `predict.py` requires `--train` and `--date` for a journey in the feature table. With no `--at`, it emits PRE for every destination. With `--at`, it uses the first matching station that has an onward segment, normalizing the code to uppercase, and predicts all subsequent arrivals. An optional `--delay` overrides `known_delay_A` and `dep_delay_A` and sets the source to departure. Leaving the override empty uses the recorded known delay. A missing recorded delay requires an override. A delay argument has an effect only with `--at`. Other context features remain those of the selected journey.
- `--csv` requires an output filename. PRE columns in order are `stop,station,sch_arrival,actual_delay,pre_delay,pre_low,pre_high`. Live inserts `stops_ahead` after `stop` and appends `live_delay,live_low,live_high,best_delay,best_low,best_high`. The CSV has no expected-arrival column. The live console table calculates expected arrival by adding the unrounded BEST delay to the scheduled arrival. Historical actuals are included when available and refer to the original journey even with an override. Use a per-request CSV output path and launch Python with the repository root as its working directory so the default paths resolve. The verified interpreter in this Windows Conda environment is `C:\GANGBANG\env\python.exe`.
- In Streamlit, the sidebar selects train, date, a current stop A and a destination B. `make_pairs(J, [i], [target_row], [0], [n])` constructs the live feature row from A's state, the path, and B's features. The segment row ending at B supplies PRE features. The form can edit these model inputs individually, handles unknown numeric inputs and categorical values, derives hour sine/cosine from scheduled departure hour, and requires a current delay. The requested web form matches the simpler CLI workflow.

## Open questions and differences from the build brief

- **Median error and bias.** The simulator emits rounded signed errors on individual checks. The dashboard uses these to calculate median absolute error and mean signed error (bias), excluding arrivals before the replay window. It displays these metrics when it has an error for every reported check. MAE, range coverage, and check count still come directly from the simulator totals.
- **Protected input directories.** If absent, the Python scripts attempt to create `_sim_journeys_<date>.csv` and `_train_index.csv` inside the selected features directory. Both caches already exist for the default August 20 replay and the default predict feature root. A UI-launched replay for another date needs a way to keep incidental cache writes outside the protected input tree. A separate feature directory is supported by the CLI.
- **Scope of the replay label.** The judge-facing August claim is supported by the bundled report and the default replay. The CLI accepts other dates, while the fixed wire note still says “test month.” If the UI exposes a date picker, should it limit the demo to August or use the August wording only for August sessions?
- **Replay identity and station display.** The message has station codes and route coordinates, with no full station names, continuous GPS, or explicit run ID. Map progress can use the latest replayed stop. The start manifest can serve as the station list for its journeys. The intended behavior for several simultaneous replay senders and the scope of train/station choices beyond the current manifest are unspecified.

`HANDOFF.md` was absent from the checkout. This review used the Python source, both payload files, and the saved model metadata and report.
