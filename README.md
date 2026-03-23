# NEXUS — Predictive Digital Twin for Data Center Servers

A minimal, end-to-end predictive digital twin prototype that models a single server rack unit. It ingests synthetic telemetry (CPU, memory, fan speed, power, temperature), trains a Gradient Boosting Regressor to predict the next-step temperature, and exposes the model through a REST API with a live monitoring dashboard.

---

## Architecture Overview

```
┌──────────────────────┐     HTTP (port 3000)     ┌───────────────────┐
│  Browser / Frontend  │ ────────────────────────▶ │   Nginx (Docker)  │
│  (Vanilla JS + CSS)  │                           │   serves HTML     │
└──────────────────────┘                           └─────────┬─────────┘
                                                             │ /api/* proxy
                                                             ▼
                                                   ┌───────────────────┐
                                                   │  FastAPI Backend   │
                                                   │  (Docker, :8000)   │
                                                   │  ┌─────────────┐  │
                                                   │  │ GBR Model   │  │
                                                   │  │ (scikit-    │  │
                                                   │  │  learn)     │  │
                                                   │  └─────────────┘  │
                                                   │  ┌─────────────┐  │
                                                   │  │ SQLite DB   │  │
                                                   │  │ (twin.db)   │  │
                                                   │  └─────────────┘  │
                                                   └───────────────────┘
```

### Components

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | HTML5 + CSS Variables + Chart.js | Live dashboard, historical charts, manual prediction |
| Web Server | Nginx (Alpine) | Serves static files, proxies `/api/*` to backend |
| Backend API | FastAPI + Uvicorn | REST endpoints for metrics, predictions, live sim |
| ML Model | scikit-learn GradientBoostingRegressor | Next-step temperature prediction |
| Database | SQLite via SQLAlchemy | Stores synthetic dataset + live readings + predictions |
| Containerisation | Docker + Docker Compose | One-command setup |

---

## Project Structure

```
digital-twin/
├── backend/
│   ├── main.py          # FastAPI app, routes, live simulation loop
│   ├── ml.py            # Dataset generator + model train/load/predict
│   ├── database.py      # SQLAlchemy models & session setup
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html       # Single-file dashboard (no build step)
├── docker-compose.yml
├── nginx.conf
└── README.md
```

---

## Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & Docker Compose v2

### Run

```bash
git clone <this-repo>
cd digital-twin
docker compose up --build
```

First run takes ~2–3 minutes (pip installs, dataset generation, model training).

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:3000 |
| **Deploy Web App** | https://digital-twin.up.railway.app/ |
| **API docs** | http://localhost:8000/docs |
| **API root** | http://localhost:8000/ |

### Stop

```bash
docker compose down
```

---

## The Digital Twin Model

### Synthetic Dataset (`ml.py:generate_dataset`)

2,000 time steps at 30-second intervals, generated with physics-inspired rules:

```
power(t)      = 80 + 1.2·cpu(t) + 0.4·mem(t) + ε
fan_target(t) = clip(20 + 2.5·(temp(t) − ambient(t)), 10, 100)
fan(t)        = 0.8·fan(t−1) + 0.2·fan_target(t)   ← lag response
dtemp(t)      = 0.003·power − 0.008·fan·ΔT − 0.002·ΔT + ε
temp(t+1)     = temp(t) + dtemp(t)
```

Where `ΔT = temp − ambient`. CPU load follows a diurnal sine wave with random 5% burst spikes.

### Predictive Model

- **Algorithm**: `GradientBoostingRegressor` (200 estimators, depth 4, lr 0.05)
- **Features**: `[cpu_util, mem_util, ambient_temp, fan_speed, power_w, temperature]`
- **Target**: `next_temperature` (temperature 30 seconds ahead)
- **Preprocessing**: `StandardScaler` in a sklearn Pipeline
- **Persistence**: serialised to `model.joblib` inside the container volume

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/stats` | Dataset statistics + model status |
| GET | `/api/metrics` | Paginated stored metrics |
| GET | `/api/metrics/recent` | Last N readings |
| GET | `/api/live` | Advance simulation one step + predict |
| POST | `/api/predict` | Ad-hoc prediction from supplied features |
| POST | `/api/metrics` | Insert a manual reading |
| GET | `/api/predictions` | Stored prediction log |
| POST | `/api/retrain` | Retrain model on all DB data |

Full interactive docs at **http://localhost:8000/docs**.

---

## Dashboard Features

- **Live telemetry gauges** — CPU, memory, fan speed, power, temperature, ambient  
- **Streaming chart** — rolling 40-point window of temp vs prediction vs CPU  
- **Prediction card** — next-step temperature with Δ vs current reading  
- **Thermal alert** — visual warning when temperature exceeds 75°C  
- **Manual prediction panel** — drag sliders and get instant inference  
- **Historical chart** — actual vs predicted temperature from the dataset  
- **Power & fan dynamics** — combined chart with dual Y axes  
- **Data tables** — live log, synthetic history, and prediction log  
- **One-click retrain** — updates model on all accumulated data  

---

## Development (without Docker)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` directly in your browser (set `API = 'http://localhost:8000'` in the script — it auto-detects localhost).

---

## Design Decisions & Scope

- **SQLite** is used for simplicity — swap `DB_URL` env var for PostgreSQL with no code changes.
- The **live simulation** runs as a stateful in-memory process inside the API, not a separate worker — keeping the architecture minimal while still being interactive.
- The **frontend is a single HTML file** with no build tooling, making it trivial to deploy anywhere.
- Model accuracy is intentionally secondary; the focus is on end-to-end integration of data → model → API → UI.

---

## Tech Choices

| Decision | Reasoning |
|----------|-----------|
| FastAPI | Async, auto-docs, Pydantic validation, fast to write |
| GradientBoostingRegressor | Strong out-of-box performance on small tabular data, no GPU needed |
| SQLAlchemy + SQLite | Zero-config relational DB, swappable |
| Chart.js 4 | Lightweight, good defaults, no build step |
| Nginx proxy | Avoids CORS issues in production; clean separation |

---

## Notes on AI Tool Usage

This project was developed in active collaboration with Claude (Anthropic) as a coding agent — used for architecture planning, boilerplate generation, debugging, and README drafting. The workflow treated the AI as a pair-programmer: prompting for structure, reviewing and correcting outputs, and iterating quickly to an end-to-end working system.
