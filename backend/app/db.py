"""Database: SQLAlchemy 2.0 models (PostgreSQL in Docker, SQLite for local development and tests)."""
from __future__ import annotations

import datetime as dt
import json

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default='analyst')            # 'admin' or 'analyst'
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Analysis(Base):
    """One analysed dataset. `summary` holds the analysis object built by app.core.analysis.build_analysis."""
    __tablename__ = 'analyses'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    dataset_name: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(20))                              # 'upload' or 'sample'
    total_flows: Mapped[int] = mapped_column(Integer)
    malicious_count: Mapped[int] = mapped_column(Integer)
    risk_level: Mapped[str] = mapped_column(String(16))
    summary: Mapped[dict] = mapped_column(JSON)
    seconds: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    flows: Mapped[list['FlowPrediction']] = relationship(back_populates='analysis', cascade='all, delete-orphan')
    reports: Mapped[list['AnalysisReport']] = relationship(back_populates='analysis', cascade='all, delete-orphan')


class FlowPrediction(Base):
    """The prediction for one flow of an analysed dataset (every flow is stored, safe and malicious)."""
    __tablename__ = 'flow_predictions'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey('analyses.id'), index=True)
    flow_index: Mapped[int] = mapped_column(Integer)                             # 1-based row number in the uploaded file
    predicted_class: Mapped[str] = mapped_column(String(32), index=True)
    malicious: Mapped[bool] = mapped_column(Boolean, index=True)
    confidence: Mapped[float] = mapped_column(Float)                              # calibrated probability of the predicted class
    confidence_level: Mapped[str] = mapped_column(String(8), index=True)          # high / medium / low
    severity: Mapped[str] = mapped_column(String(10), index=True)                 # Critical / High / Medium / Low / None
    prediction_set: Mapped[list] = mapped_column(JSON)                           # 90% conformal shortlist
    ground_truth: Mapped[str | None] = mapped_column(String(32), nullable=True)  # only when the file carries labels
    flow: Mapped[dict] = mapped_column(JSON)                                     # the 42 raw columns
    analysis: Mapped[Analysis] = relationship(back_populates='flows')
    incident_report: Mapped['FlowReport | None'] = relationship(back_populates='flow_prediction', cascade='all, delete-orphan', uselist=False)


class FlowReport(Base):
    """Optional verified incident explanation for one flow (local LLM + MITRE ATT&CK / NIST / CISA sources)."""
    __tablename__ = 'flow_reports'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    flow_id: Mapped[int] = mapped_column(ForeignKey('flow_predictions.id'), unique=True)
    content: Mapped[dict] = mapped_column(JSON)
    llm_used: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    flow_prediction: Mapped[FlowPrediction] = relationship(back_populates='incident_report')


class AnalysisReport(Base):
    """A generated PDF security report of one analysis (the file lives under <data_dir>/reports)."""
    __tablename__ = 'analysis_reports'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey('analyses.id'), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    pages: Mapped[int] = mapped_column(Integer, default=0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    rag: Mapped[dict | None] = mapped_column(JSON, nullable=True)                 # schema 1: threat intelligence (RAG + LLM, verified)
    record: Mapped[dict | None] = mapped_column(JSON, nullable=True)              # schema 2: the full versioned report record (web + PDF + exports)
    schema_version: Mapped[str | None] = mapped_column(String(10), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    analysis: Mapped[Analysis] = relationship(back_populates='reports')


class ActionReview(Base):
    """Analyst checklist for the recommended actions of a report. Every change is a new row (full history); the latest row is the state.
    The application never executes an action: these states only record what an analyst reviewed, planned or did."""
    __tablename__ = 'action_reviews'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey('analysis_reports.id'), index=True)
    action_id: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(24))                              # open / reviewed / planned / completed_by_analyst / not_applicable
    note: Mapped[str] = mapped_column(Text, default='')
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class AuditLog(Base):
    __tablename__ = 'audit_log'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey('users.id'), nullable=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    detail: Mapped[str] = mapped_column(Text, default='')
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


def make_session_factory(url: str):
    kwargs = {'connect_args': {'check_same_thread': False}} if url.startswith('sqlite') else {'pool_pre_ping': True}
    engine = create_engine(url, json_serializer=lambda o: json.dumps(o, default=float), **kwargs)
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    return engine, sessionmaker(engine, expire_on_commit=False)


def _add_missing_columns(engine):
    """Tiny migration for databases created by an earlier version (create_all does not add columns to existing tables)."""
    existing = {c['name'] for c in inspect(engine).get_columns('analysis_reports')}
    for column, kind in (('rag', 'JSON'), ('record', 'JSON'), ('schema_version', 'VARCHAR(10)')):
        if column not in existing:
            with engine.begin() as connection:
                connection.execute(text(f'ALTER TABLE analysis_reports ADD COLUMN {column} {kind}'))
