"""
Predictive Digital Twin — FastAPI Backend
"""
from __future__ import annotations

import os
import math
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db, init_db, ServerMetric, Prediction
from ml import (
    generate_dataset,
    train_model,
    load_model,
    predict,
    FEATURES,
    MODEL_PATH,
)

# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _bootstrap()
    yield


def _bootstrap():
    """Seed DB and train model if not already done."""
    from sqlalchemy.orm import Session as S
    from database import SessionLocal

    db: S = SessionLocal()
    try:
        count = db.query(ServerMetric).count()
        if count == 0:
            print("Seeding database with synthetic data …")
            df = generate_dataset()
            records = [
                ServerMetric(
                    timestamp=row.timestamp.to_pydatetime(),
                    cpu_util=row.cpu_util,
                    mem_util=row.mem_util,
                    ambient_temp=row.ambient_temp,
                    fan_speed=row.fan_speed,
                    power_w=row.power_w,
                    temperature=row.temperature,
                    next_temperature=row.next_temperature,
                    source="synthetic",
                )
                for _, row in df.iterrows()
            ]
            db.bulk_save_objects(records)
            db.commit()
            print(f"Inserted {len(records)} rows.")
        else:
            print(f"DB already has {count} rows, skipping seed.")

        def _do_train():
            from database import SessionLocal as SL
            import pandas as pd
            _db = SL()
            rows = _db.query(ServerMetric).filter(
                ServerMetric.source == "synthetic"
            ).all()
            _db.close()
            df = pd.DataFrame([
                {
                    "cpu_util": r.cpu_util,
                    "mem_util": r.mem_util,
                    "ambient_temp": r.ambient_temp,
                    "fan_speed": r.fan_speed,
                    "power_w": r.power_w,
                    "temperature": r.temperature,
                    "next_temperature": r.next_temperature,
                }
                for r in rows if r.next_temperature is not None
            ])
            train_model(df)

        if not os.path.exists(MODEL_PATH):
            print("Training predictive model …")
            _do_train()
            print("Model trained and saved.")
        else:
            # Try loading; if it fails (version mismatch), retrain fresh
            try:
                app.state.model = load_model()
                print("Model loaded successfully.")
                return
            except Exception as e:
                print(f"Model load failed ({e}), retraining …")
                os.remove(MODEL_PATH)
                _do_train()
                print("Model retrained and saved.")

        app.state.model = load_model()
        print("Model ready.")
    finally:
        db.close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Server Digital Twin API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class MetricIn(BaseModel):
    cpu_util: float
    mem_util: float
    ambient_temp: float
    fan_speed: float
    power_w: float
    temperature: float


class MetricOut(MetricIn):
    id: int
    timestamp: datetime
    next_temperature: Optional[float] = None
    source: str

    class Config:
        from_attributes = True


class PredictionOut(BaseModel):
    predicted_temperature: float
    timestamp: datetime

    class Config:
        from_attributes = True


class PredictRequest(BaseModel):
    cpu_util: float
    mem_util: float
    ambient_temp: float
    fan_speed: float
    power_w: float
    temperature: float


class LiveStateOut(BaseModel):
    cpu_util: float
    mem_util: float
    ambient_temp: float
    fan_speed: float
    power_w: float
    temperature: float
    predicted_temperature: float
    timestamp: datetime


class StatsOut(BaseModel):
    total_records: int
    avg_temperature: float
    avg_cpu_util: float
    avg_power_w: float
    max_temperature: float
    model_ready: bool


# ── Live simulation state ─────────────────────────────────────────────────────

_rng = np.random.default_rng(99)
_live_state = {
    "cpu_util": 45.0,
    "mem_util": 55.0,
    "ambient_temp": 22.0,
    "fan_speed": 50.0,
    "power_w": 200.0,
    "temperature": 52.0,
}


def _tick_live_state():
    """Advance the live simulation one step."""
    s = _live_state
    s["cpu_util"] = float(np.clip(s["cpu_util"] + _rng.normal(0, 3), 5, 100))
    s["mem_util"] = float(np.clip(s["mem_util"] + _rng.normal(0, 1.5), 10, 95))
    s["ambient_temp"] = float(np.clip(s["ambient_temp"] + _rng.normal(0, 0.1), 18, 28))

    power = 80 + 1.2 * s["cpu_util"] + 0.4 * s["mem_util"] + float(_rng.normal(0, 3))
    s["power_w"] = float(np.clip(power, 80, 450))

    target_fan = np.clip(20 + 2.5 * (s["temperature"] - s["ambient_temp"]), 10, 100)
    s["fan_speed"] = float(np.clip(0.8 * s["fan_speed"] + 0.2 * target_fan + _rng.normal(0, 1), 10, 100))

    heat_in = 0.003 * s["power_w"]
    cool_fan = 0.008 * s["fan_speed"] * (s["temperature"] - s["ambient_temp"])
    cool_amb = 0.002 * (s["temperature"] - s["ambient_temp"])
    dtemp = heat_in - cool_fan - cool_amb + float(_rng.normal(0, 0.2))
    s["temperature"] = float(np.clip(s["temperature"] + dtemp, s["ambient_temp"] + 5, 95))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "Server Digital Twin API"}


@app.get("/api/stats", response_model=StatsOut)
def get_stats(db: Session = Depends(get_db)):
    from sqlalchemy import func
    total = db.query(ServerMetric).count()
    agg = db.query(
        func.avg(ServerMetric.temperature),
        func.avg(ServerMetric.cpu_util),
        func.avg(ServerMetric.power_w),
        func.max(ServerMetric.temperature),
    ).one()
    return StatsOut(
        total_records=total,
        avg_temperature=round(agg[0] or 0, 2),
        avg_cpu_util=round(agg[1] or 0, 2),
        avg_power_w=round(agg[2] or 0, 2),
        max_temperature=round(agg[3] or 0, 2),
        model_ready=os.path.exists(MODEL_PATH),
    )


@app.get("/api/metrics", response_model=List[MetricOut])
def get_metrics(
    limit: int = Query(200, le=2000),
    offset: int = Query(0),
    source: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(ServerMetric).order_by(ServerMetric.timestamp)
    if source:
        q = q.filter(ServerMetric.source == source)
    return q.offset(offset).limit(limit).all()


@app.get("/api/metrics/recent", response_model=List[MetricOut])
def get_recent_metrics(
    n: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(ServerMetric)
        .order_by(ServerMetric.id.desc())
        .limit(n)
        .all()
    )
    return list(reversed(rows))


@app.post("/api/predict", response_model=dict)
def predict_endpoint(req: PredictRequest):
    model = app.state.model
    feat = {
        "cpu_util": req.cpu_util,
        "mem_util": req.mem_util,
        "ambient_temp": req.ambient_temp,
        "fan_speed": req.fan_speed,
        "power_w": req.power_w,
        "temperature": req.temperature,
    }
    pred = predict(model, feat)
    return {"predicted_temperature": round(pred, 2)}


@app.get("/api/live", response_model=LiveStateOut)
def get_live(db: Session = Depends(get_db)):
    _tick_live_state()
    s = _live_state.copy()
    model = app.state.model
    pred = predict(model, s)

    # Persist live reading
    rec = ServerMetric(
        timestamp=datetime.utcnow(),
        source="live",
        **s,
        next_temperature=round(pred, 2),
    )
    db.add(rec)

    pred_rec = Prediction(
        timestamp=datetime.utcnow(),
        predicted_temperature=round(pred, 2),
        **s,
    )
    db.add(pred_rec)
    db.commit()

    return LiveStateOut(
        **s,
        predicted_temperature=round(pred, 2),
        timestamp=datetime.utcnow(),
    )


@app.get("/api/predictions", response_model=List[dict])
def get_predictions(
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Prediction)
        .order_by(Prediction.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp,
            "temperature": r.temperature,
            "predicted_temperature": r.predicted_temperature,
            "cpu_util": r.cpu_util,
            "fan_speed": r.fan_speed,
            "power_w": r.power_w,
        }
        for r in reversed(rows)
    ]


@app.post("/api/metrics", response_model=MetricOut)
def add_metric(metric: MetricIn, db: Session = Depends(get_db)):
    model = app.state.model
    feat = metric.model_dump()
    pred = predict(model, feat)
    rec = ServerMetric(
        **feat,
        next_temperature=round(pred, 2),
        source="manual",
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


@app.post("/api/retrain")
def retrain(db: Session = Depends(get_db)):
    import pandas as pd
    rows = db.query(ServerMetric).filter(
        ServerMetric.next_temperature.isnot(None)
    ).all()
    if len(rows) < 100:
        raise HTTPException(400, "Not enough data to retrain (need >= 100 rows).")
    df = pd.DataFrame([
        {
            "cpu_util": r.cpu_util,
            "mem_util": r.mem_util,
            "ambient_temp": r.ambient_temp,
            "fan_speed": r.fan_speed,
            "power_w": r.power_w,
            "temperature": r.temperature,
            "next_temperature": r.next_temperature,
        }
        for r in rows
    ])
    model = train_model(df)
    app.state.model = model
    return {"status": "retrained", "rows_used": len(df)}