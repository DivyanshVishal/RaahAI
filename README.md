# RaahAI

RaahAI forecasts arrival delays at upcoming Indian railway stations. The web route finder shows scheduled and recorded stops on a map, alongside predictions and delay ranges from pretrained Python LightGBM models. The live-feed demo replays recorded journeys to show how forecasts change as station reports arrive.

## Run locally on Windows

Install Node.js and Python. From the repository root, create the Python environment and install the dependencies.

```powershell
python -m venv env
.\env\Scripts\python.exe -m pip install -r requirements.txt lightgbm
```

Provide the prepared journey data in `processed/` and pretrained model artifacts in `models/`. These local artifacts are excluded from Git and are needed to reproduce the full forecast demo. See [the web setup guide](web/README.md) for data preparation and set `PYTHON_PATH` in `web/.env.local` to the absolute path of your Python interpreter.

```powershell
cd web
npm.cmd ci
npm.cmd run dev
```

Open [the route finder](http://localhost:3000). The Next.js server also serves the backend APIs.

## Replay and evaluation

Open [the live-feed guide](live_feed/README.md) to run the recorded station feed. The included [algorithm benchmark](train_delay_algorithm_benchmark.ipynb) compares candidate models on a sample and has its own dependencies in `requirements-benchmark.txt`. The application uses pretrained Python models for inference.

See [the simulator message guide](docs/CODEX_UNDERSTANDING.md) for the event formats.
