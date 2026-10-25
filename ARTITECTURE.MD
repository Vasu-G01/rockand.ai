# AstroLand AI — System Architecture

## 1. Overview

AstroLand AI is an intelligent rocket landing simulator designed to combine interactive visualization, physics-based simulation, and machine learning.

The project follows a frontend-backend architecture. The frontend handles the user interface and rocket visualization, while the Python backend manages simulation logic, ML predictions, and data storage.

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React.js, JavaScript |
| Graphics | HTML Canvas |
| Styling | CSS |
| Backend | Python, Flask |
| API Communication | Flask-CORS |
| Machine Learning | Scikit-learn, NumPy, Pandas |
| Database | SQLite |

## 3. Project Structure

```text
AstroLand-AI/
├── frontend/
│   ├── public/
│   │   └── assets/
│   └── src/
│       ├── App.jsx              # canvas simulation + AI pilot integration
│       ├── api.js               # backend client + local guidance fallback
│       ├── index.css
│       ├── App.css
│       └── main.jsx
│   └── package.json
│
├── backend/
│   ├── app.py                   # Flask application and API endpoints
│   ├── config.py                # paths, ports, training defaults
│   ├── requirements.txt
│   ├── simulation/
│   │   ├── physics.py           # port of the canvas physics (shared rules)
│   │   └── landing.py           # touchdown evaluation + scoring
│   ├── ml/
│   │   ├── env.py               # Gym environment used for training
│   │   ├── train_model.py       # PPO training, evaluation, status file
│   │   ├── predict.py           # Pilot: load policy, act(), plan()
│   │   └── models/              # ppo_rocket_land.zip, meta + status JSON
│   ├── database/
│   │   └── database.py          # SQLite flights table
│   └── tests/
│       ├── test_physics.py
│       └── test_prediction.py
│
├── .gitignore
├── README.md
└── ARTITECTURE.MD
```

## 4. Component Responsibilities

### Frontend
- **App.jsx:** Rocket simulation canvas, manual controls and the AI pilot loop
  (requests a control every 100 ms, re-plans the path, draws the predicted path).
- **api.js:** HTTP client for the Flask API plus a local guidance controller used
  when the backend is unreachable.
- **index.css:** all styling.

### Backend
- **app.py:** Flask API — `/api/predict`, `/api/plan`, `/api/train*`, `/api/flights`.
- **config.py:** model paths, ports and training defaults.
- **physics.py:** the single source of truth for the rocket dynamics (kept in
  lockstep with App.jsx) and the shared spawn ranges.
- **landing.py:** LANDED vs CRASHED evaluation and scoring.
- **env.py:** Gym environment (`RocketLandEnv`, Discrete(6) actions, 10 features)
  with a spawn curriculum: 0 = close to the pad, 2 = exact browser randomisation.
- **train_model.py:** trains PPO, streams progress to `training_status.json`,
  evaluates the policy and records the metadata.
- **predict.py:** `Pilot` singleton — loads the trained policy, answers single
  control decisions (`act`) and full predicted flights (`plan`), with a
  classical guidance controller as fallback.
- **database.py:** SQLite storage for finished flights.

### Tests
- **test_physics.py:** dynamics, limits, spawn ranges and scoring.
- **test_prediction.py:** features, guidance law, environment and pilot output.

## 5. System Workflow

### Manual flight
1. The user interacts with the React frontend and the canvas renders the rocket.
2. On landing or crash the result is stored locally and posted to `/api/flights`.

### AI pilot (trained model)
1. The user presses **🤖 AI pilot: ON** (manual keys still override at any time).
2. The frontend posts the telemetry to `/api/plan`; the backend rolls the trained
   policy forward and returns the predicted path, which is drawn as a dashed line.
3. Every 100 ms the frontend posts the telemetry to `/api/predict` and applies the
   returned control (thrust / rotate left / rotate right) — one decision at a time.
4. The path is re-planned every few seconds, so the pilot corrects any drift.
5. Results are saved to SQLite; training progress and model info are shown in the
   "ML autopilot" panel.

### Training
1. `ml/train_model.py` builds 4 parallel environments and trains PPO.
2. Progress is written to `ml/models/training_status.json`, which the API exposes
   through `/api/train/status`.
3. On completion the policy (`ppo_rocket_land.zip`) is evaluated and the API
   hot-reloads it, so the next flight uses the new model.

### API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | model + training + database overview |
| GET | `/api/model` | loaded policy info |
| POST | `/api/model/reload` | re-read the model file |
| POST | `/api/predict` | one control decision for a telemetry state |
| POST | `/api/plan` | predicted flight path + control sequence |
| POST | `/api/train` | start training (background process) |
| GET | `/api/train/status` | live training progress |
| POST | `/api/train/stop` | graceful stop |
| GET/POST | `/api/flights` | flight history |

## 6. Team Development Guidelines

- Keep frontend and backend code in their respective folders.
- Create a separate feature branch for each task.
- Test changes before opening a Pull Request.
- Keep dependencies documented in `package.json` and `requirements.txt`.
- Never commit secrets or virtual environment folders.
- Update this document when the architecture changes.

## 7. Project Status

Under Development. The structure describes the planned architecture; individual components may not be implemented yet.