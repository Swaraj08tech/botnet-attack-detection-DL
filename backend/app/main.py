"""FastAPI application: loads the serving bundle once at start-up and exposes the API under /api."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select

from app.config import Settings
from app.config import settings as default_settings
from app.core.datasets import UploadStore
from app.core.detector import Detector
from app.core.reports import ReportEngine
from app.db import User, make_session_factory
from app.jobs import JobRunner
from app.routers import ANALYSIS_STAGES, INCIDENT_STAGES, REPORT_STAGES, router
from app.security import hash_password

log = logging.getLogger('botnet-api')


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or default_settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine, app.state.sessions = make_session_factory(cfg.database_url)
        app.state.detector = Detector(cfg.bundle_dir)
        app.state.reports = ReportEngine(cfg.bundle_dir, cfg.ollama_url, cfg.llm_model, app.state.detector.predictors)
        app.state.analysis_jobs = JobRunner('analysis', ANALYSIS_STAGES)
        app.state.report_jobs = JobRunner('report', REPORT_STAGES)
        app.state.incident_jobs = JobRunner('incident', INCIDENT_STAGES)
        app.state.data_dir = cfg.data_dir
        app.state.uploads = UploadStore(cfg.data_dir / 'uploads')
        descriptions = cfg.bundle_dir / 'feature_descriptions.json'
        app.state.feature_descriptions = json.loads(descriptions.read_text(encoding='utf-8')) if descriptions.exists() else {}

        def optional_json(name):
            path = cfg.bundle_dir / name
            return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        app.state.knowledge_sources = optional_json('knowledge_sources.json')      # source register (titles, versions, URLs, hashes)
        app.state.feature_reference = optional_json('feature_reference.json')      # percentiles of the fitting rows
        app.state.class_reference = optional_json('class_reference.json')          # how typical a value is per class (fitting rows)
        app.state.demo = pd.read_csv(cfg.bundle_dir / 'demo_flows.csv', low_memory=False)
        app.state.model_card = json.loads((cfg.bundle_dir / 'model_card.json').read_text(encoding='utf-8'))
        with app.state.sessions() as db:                     # first start: create the administrator account
            if db.scalar(select(func.count()).select_from(User)) == 0:
                if not cfg.admin_password:
                    raise RuntimeError('No users exist yet: set ADMIN_EMAIL and ADMIN_PASSWORD for the first administrator')
                db.add(User(email=cfg.admin_email.lower(), name='Administrator', password_hash=hash_password(cfg.admin_password), role='admin'))
                db.commit()
                log.info('created first administrator %s', cfg.admin_email)
        yield
        for runner in (app.state.analysis_jobs, app.state.report_jobs, app.state.incident_jobs):
            runner.shutdown()
        app.state.engine.dispose()

    app = FastAPI(title='Network Attack Detection API', version='2.0.0', lifespan=lifespan,
                  description='Detector v2 (LightGBM, UNSW-NB15 features) with calibrated confidence, conformal prediction sets, '
                              'TreeSHAP explanations and verified incident reports.')
    app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    app.include_router(router)
    return app


app = create_app()
