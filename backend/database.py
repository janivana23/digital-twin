"""SQLite database setup using SQLAlchemy."""
import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, DateTime, String, Text
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.pool import StaticPool

DB_URL = os.environ.get("DB_URL", "sqlite:///./twin.db")

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {},
    poolclass=StaticPool if "sqlite" in DB_URL else None,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class ServerMetric(Base):
    __tablename__ = "server_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cpu_util = Column(Float, nullable=False)
    mem_util = Column(Float, nullable=False)
    ambient_temp = Column(Float, nullable=False)
    fan_speed = Column(Float, nullable=False)
    power_w = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    next_temperature = Column(Float, nullable=True)
    source = Column(String(32), default="synthetic")  # "synthetic" | "live"


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cpu_util = Column(Float)
    mem_util = Column(Float)
    ambient_temp = Column(Float)
    fan_speed = Column(Float)
    power_w = Column(Float)
    temperature = Column(Float)
    predicted_temperature = Column(Float)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
