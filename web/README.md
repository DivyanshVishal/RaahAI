# RaahAI train route finder

Search the landing page by train name or number. An exact number opens the route directly. Broader searches list matching trains. The initial example is Ranchi Express (18636). Each train opens on its latest available journey, and the recorded journey selector lists its other dates.

The station timetable shows scheduled arrival and departure, recorded arrival and delay, platform, and distance. Separate columns show the pre-departure delay forecast and its low–high range from the pretrained Python models. The timetable matches forecasts to the station and scheduled arrival on the selected journey. All times are in IST. Select a station code to locate it on the map. The line connects known station coordinates in stop order. A dash indicates a missing value. The first and last stops are the endpoints present in the selected record.

## Run on Windows

From the repository root with Node.js installed:

```powershell
cd web
npm.cmd install
npm.cmd run dev
```

Open [the route finder](http://localhost:3000). The app reads the prepared 2026 data with the repository's Python environment. To use another Python interpreter with PyArrow installed, set `PYTHON_PATH` in `web/.env.local` and restart Next.js. OpenStreetMap background tiles require internet access.

## Data and verification

Train search uses `web/data/train-catalog.json`, built from the feature train index and the names in the clean Parquet partitions. Each route and available date list comes from the selected train's clean partition. The map uses the station coordinate lookup with the stop's recorded GPS coordinates as a fallback.

To regenerate the search catalog after updating the prepared dataset, run from the repository root:

```powershell
.\env\python.exe -B web/scripts/train_routes.py catalog
```

Restart the app after regeneration. Route reads are cached for up to an hour. Run `npm.cmd run typecheck` and `npm.cmd run build` from `web` to check the application.

## Simulator and prediction APIs

The route finder runs independently of the simulator. The existing endpoints remain available for programmatic use.

- `POST /api/predictions` accepts simulator messages. `GET /api/state` returns the current snapshot, and `GET /api/stream` streams updates.
- `POST /api/predict` runs the Python prediction CLI for a train and date, with optional station and delay.
- To send replay events to the backend, run `python simulate_live.py --start-time 07:00 --hours 3` from the repository root in the project environment. Its default log path is `sim_log.jsonl`. To post the existing fixture, run `npm.cmd run replay` from `web`.
