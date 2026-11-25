# Recorded station feed

From the repository root on Windows, start one of the following runs:

```powershell
.\env\python.exe -B live_feed\server.py
.\env\python.exe -B live_feed\server.py --split validation --train 18636 --date 2026-07-21 --speed 60
```

Open [the feed viewer](http://127.0.0.1:8100/). The default run serves train 18636 on August 20 at 120 simulated seconds per real second, with long gaps between reports capped at five real seconds. Shorter gaps keep their scaled timing, and reports with the same release time appear together. Use `--max-gap 0` for uncapped playback at the selected speed.

The first event request starts a single shared clock. Refreshing the viewer shows the published history. To restart or change journey, stop the process with Ctrl+C and run the desired command. Use `--port` to run an additional feed.

## Consumer API

Poll `GET /api/events?after=0` for all published events. Pass the returned `cursor` as `after` on the next poll. An empty event list means no new observations have been released. Stop polling when `complete` is true. Each client maintains its own cursor.

For a replay of a specific journey, include `train_no`, `journey_date` (YYYY-MM-DD), and a client-generated UUID in `session`. Keep those values and the returned `run` on subsequent requests. Use a new UUID to restart the simulation.

Each batch identifies the train, journey date, split, playback speed and maximum gap in real seconds. Each event contains its ID, arrival or departure type, station code, zero-based stop index, scheduled time, actual time, availability time and recorded delay in minutes. Timestamps include the IST offset. The server reads the cleaned station records and releases an observation only once both its actual time and station report time have passed on the simulated clock. Stops without a timestamped report are skipped.

The split follows the project's journey-start-date rule. July 2026 is validation and August 2026 is test. Overnight stops stay with their originating journey.

## See the updated forecasts

For the presentation journey, run this feed from the repository root:

```powershell
.\env\python.exe -B live_feed\server.py --train 73482 --date 2026-08-07 --speed 30
```

Open [the matching route](http://localhost:3000/?train=73482&date=2026-08-07) in the route finder and click **Connect simulated live feed**. The website starts a fresh simulation for the train and date you selected and can replay any journey with recorded July or August observations. The simulated clock keeps moving during gaps between reports. If the feed runs on a custom port, set `LIVE_FEED_URL` in the web app environment.

The API also publishes `forecasts` and `scores`. Forecasts use the saved models and only reports released through that point in the simulation. Scores compare the latest forecast that was available before the actual arrival with the pre-departure estimate for the same stop. Use `--observations-only` for a station feed without model inference.
