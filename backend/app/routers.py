"""All API routes, under /api. Every route except /api/health and /api/auth/login needs a valid token.

The main workflow:
  POST /uploads (or /uploads/sample)   upload + validate a dataset -> upload id and a validation report
  POST /analyses                       start the analysis of a validated upload -> job with real stages
  GET  /analyses/jobs/{job}            stage of the analysis job; when done: the analysis id
  GET  /analyses/{id}                  the analysis object (Results page, charts, report)
  GET  /analyses/{id}/flows            paginated, filterable flow predictions
  POST /analyses/{id}/report           start the PDF security report -> job with real stages
  GET  /reports, /reports/{id}/pdf     generated reports
Plus: flow detail with explanation and the optional verified incident explanation (local LLM), overview statistics, model information.
"""
from __future__ import annotations

import hashlib
import io
import json
import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import String, cast, func, insert, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.config import settings
from app.core import analysis as rules
from app.core.analysis import build_analysis, flow_fields, status_confidence, status_summary, true_labels
from app.core.datasets import validate_dataset
from app.core.detector import CLASSES, InputError
from app.core.pdf_report import render_record_pdf, render_summary_pdf
from app.core.report_record import RecordBuilder
from app.core.report_views import views as report_views
from app.db import ActionReview, Analysis, AnalysisReport, FlowPrediction, FlowReport, User
from app.jobs import UserFacingError
from app.security import audit, create_token, current_user, get_db, login_limiter, verify_password

router = APIRouter(prefix='/api')

ANALYSIS_STAGES = ['features', 'detection', 'confidence', 'analytics']
REPORT_STAGES = ['detections', 'explanations', 'retrieval', 'generation', 'checks', 'pdf']
ACTION_STATES = ('open', 'reviewed', 'planned', 'completed_by_analyst', 'not_applicable')
INCIDENT_STAGES = ['evidence', 'writing', 'verifying', 'repairing', 'saving']


class AnalyzeRequest(BaseModel):
    upload_id: str = Field(min_length=8, max_length=64)


class PredictRequest(BaseModel):
    flows: list[dict] = Field(min_length=1, max_length=10_000)


def user_out(u: User) -> dict:
    return {'id': u.id, 'email': u.email, 'name': u.name, 'role': u.role}


def iso(value) -> str:
    """Times are stored in UTC; always send them with the zone so the browser shows local time."""
    return value.isoformat() if value.tzinfo else value.isoformat() + '+00:00'


# ------------------------------------------------------------------------------------------------------------ health + auth
@router.get('/health')
def health(request: Request):
    state = request.app.state
    return {'status': 'ok', 'model': state.detector.manifest.get('detector', 'unknown') if state.detector else None,
            'model_loaded': state.detector is not None, 'report_generator': True,
            'llm_available': state.reports.available() if state.reports else False}


@router.post('/auth/login')
def login(request: Request, form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    key = request.client.host if request.client else 'unknown'
    login_limiter.check(key)
    user = db.scalar(select(User).where(User.email == form.username.lower()))
    if user is None or not user.active or not verify_password(form.password, user.password_hash):
        login_limiter.fail(key)
        audit(db, None, 'login_failed', form.username[:255])
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, 'Wrong email or password')
    audit(db, user, 'login')
    return {'access_token': create_token(user), 'token_type': 'bearer', 'user': user_out(user)}


@router.get('/auth/me')
def me(user: User = Depends(current_user)):
    return user_out(user)


# ------------------------------------------------------------------------------------------------------------ 1. upload + validate
def stage_upload(request: Request, content: bytes, filename: str) -> dict:
    state = request.app.state
    report, _ = validate_dataset(content, filename, state.detector, settings.max_upload_rows, state.feature_descriptions)
    report['upload_id'] = state.uploads.save(content, report) if report['valid'] else None
    return report


@router.post('/uploads')
async def upload_dataset(request: Request, file: UploadFile = File(...), _: User = Depends(current_user)):
    """Upload a CSV and validate it against the model's requirements. Nothing is predicted yet."""
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f'The file is larger than {settings.max_upload_mb} MB.')
    return stage_upload(request, content, file.filename or 'upload.csv')


@router.post('/uploads/sample')
def upload_sample(request: Request, per_class: int = Query(30, ge=1, le=300), _: User = Depends(current_user)):
    """Stage the built-in sample: real flows from the official test file, `per_class` of each class (true class in `ground_truth`)."""
    return stage_upload(request, sample_frame(request, per_class).to_csv(index=False).encode(), f'unsw_nb15_sample_{per_class * len(CLASSES)}.csv')


def sample_frame(request: Request, per_class: int) -> pd.DataFrame:
    demo = request.app.state.demo
    parts = [g.sample(min(per_class, len(g)), random_state=7) for _, g in demo.groupby('ground_truth')]
    return pd.concat(parts).sample(frac=1, random_state=7).reset_index(drop=True)


@router.get('/samples/flows.csv')
def sample_csv(request: Request, per_class: int = Query(30, ge=1, le=300), _: User = Depends(current_user)):
    sample = sample_frame(request, per_class)
    return Response(sample.to_csv(index=False), media_type='text/csv',
                    headers={'Content-Disposition': f'attachment; filename="unsw_nb15_sample_{len(sample)}.csv"'})


@router.delete('/uploads/{upload_id}', status_code=204)
def discard_upload(upload_id: str, request: Request, _: User = Depends(current_user)):
    request.app.state.uploads.delete(upload_id)


# ------------------------------------------------------------------------------------------------------------ 2. analyse (background job)
def run_analysis(app, upload_id: str, user_id: int, progress) -> dict:
    staged = app.state.uploads.load(upload_id)
    if staged is None:
        raise UserFacingError('The uploaded file is no longer available. Please upload it again.')
    content, report = staged
    start = time.time()
    detector = app.state.detector
    frame = pd.read_csv(io.BytesIO(content), low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    truth = true_labels(frame)

    progress('features')                                   # validate + one-hot + engineered features (the model matrix)
    try:
        x = detector.transform(frame)
    except InputError as error:
        raise UserFacingError(f'The dataset could not be prepared for the model: {error}') from error
    progress('detection')                                  # LightGBM
    raw = detector.raw_proba(x)
    progress('confidence')                                 # temperature calibration + conformal prediction sets
    proba = detector.calibrate(raw)
    sets = detector.prediction_sets(proba)
    predicted = proba.argmax(axis=1)
    confidence = proba[np.arange(len(proba)), predicted]

    progress('analytics')                                  # the analysis object + every flow prediction stored
    summary = build_analysis(predicted, confidence, sets, truth, status_confidence(proba, predicted))
    summary['input'] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    fields = flow_fields(predicted, confidence, sets)
    flows = frame[detector.predictors].to_dict('records')
    with app.state.sessions() as db:
        source = 'sample' if report['filename'].startswith('unsw_nb15_sample_') else 'upload'
        item = Analysis(user_id=user_id, dataset_name=report['filename'][:255], source=source,
                        total_flows=summary['total_flows'], malicious_count=summary['malicious_count'], risk_level=summary['risk']['level'],
                        summary=summary, seconds=0.0)
        db.add(item)
        db.flush()
        rows = [{'analysis_id': item.id, 'flow_index': i + 1, 'predicted_class': str(fields['classes'][i]), 'malicious': bool(fields['malicious'][i]),
                 'confidence': float(confidence[i]), 'confidence_level': str(fields['level'][i]), 'severity': str(fields['severity'][i]),
                 'prediction_set': [CLASSES[j] for j in np.flatnonzero(sets[i])],
                 'ground_truth': None if truth is None or truth[i] not in CLASSES else str(truth[i]), 'flow': flows[i]}
                for i in range(len(frame))]
        for chunk in range(0, len(rows), 5000):
            db.execute(insert(FlowPrediction), rows[chunk:chunk + 5000])
        item.seconds = round(time.time() - start, 2)
        db.commit()
        audit(db, db.get(User, user_id), 'analysis', f'{item.dataset_name}: {item.total_flows} flows, {item.malicious_count} malicious')
        analysis_id = item.id
    app.state.uploads.delete(upload_id)
    return {'analysis_id': analysis_id}


@router.post('/analyses', status_code=202)
def start_analysis(body: AnalyzeRequest, request: Request, user: User = Depends(current_user)):
    if request.app.state.uploads.load(body.upload_id) is None:
        raise HTTPException(404, 'Upload not found or expired. Please upload the file again.')
    runner = request.app.state.analysis_jobs
    job = runner.submit(f'analysis:{body.upload_id}', lambda progress: run_analysis(request.app, body.upload_id, user.id, progress))
    return runner.describe(job)


@router.get('/analyses/jobs/{job_id}')
def analysis_job(job_id: str, request: Request, _: User = Depends(current_user)):
    runner = request.app.state.analysis_jobs
    job = runner.get(job_id)
    if job is None:
        raise HTTPException(404, 'Analysis job not found')
    return runner.describe(job)


# ------------------------------------------------------------------------------------------------------------ 3. results
def analysis_out(a: Analysis, full: bool = True) -> dict:
    latest = max(a.reports, key=lambda r: r.id) if a.reports else None
    out = {'id': a.id, 'dataset_name': a.dataset_name, 'source': a.source, 'created_at': iso(a.created_at), 'seconds': a.seconds,
           'total_flows': a.total_flows, 'malicious_count': a.malicious_count, 'safe_count': a.total_flows - a.malicious_count,
           'risk_level': a.risk_level, 'report': report_out(latest) if latest else None}
    if full:
        out['analysis'] = a.summary
    else:
        out['malicious_percentage'] = a.summary.get('malicious_percentage', 0)
        out['average_confidence'] = a.summary.get('average_confidence', 0)
        out['has_labels'] = 'evaluation' in a.summary
    return out


def get_analysis(db: Session, analysis_id: int) -> Analysis:
    item = db.get(Analysis, analysis_id)
    if item is None:
        raise HTTPException(404, 'Analysis not found')
    return item


@router.get('/analyses')
def list_analyses(limit: int = Query(100, le=500), db: Session = Depends(get_db), _: User = Depends(current_user)):
    return [analysis_out(a, full=False) for a in db.scalars(select(Analysis).order_by(Analysis.id.desc()).limit(limit))]


def ensure_status_confidence(app, db: Session, item: Analysis) -> None:
    """Analyses created before the attack-or-normal confidence existed get it once, from their stored flows and the same detector
    (identical predictions; only the summary gains the new counts)."""
    if item.summary is None or 'status_confidence_distribution' in item.summary:
        return
    rows = db.scalars(select(FlowPrediction.flow).where(FlowPrediction.analysis_id == item.id).order_by(FlowPrediction.flow_index)).all()
    if not rows:
        return
    pred = app.state.detector.predict(pd.DataFrame(rows))
    item.summary = {**item.summary, **status_summary(status_confidence(pred.proba, pred.predicted))}
    flag_modified(item, 'summary')
    db.commit()


@router.get('/analyses/{analysis_id}')
def analysis_detail(analysis_id: int, request: Request, db: Session = Depends(get_db), _: User = Depends(current_user)):
    item = get_analysis(db, analysis_id)
    ensure_status_confidence(request.app, db, item)
    return analysis_out(item)


@router.delete('/analyses/{analysis_id}', status_code=204)
def delete_analysis(analysis_id: int, request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = get_analysis(db, analysis_id)
    for r in item.reports:
        (request.app.state.data_dir / 'reports' / r.filename).unlink(missing_ok=True)
    db.delete(item)
    db.commit()
    audit(db, user, 'analysis_deleted', f'#{analysis_id}')


def flow_out(f: FlowPrediction, full: bool = False) -> dict:
    out = {'id': f.id, 'analysis_id': f.analysis_id, 'flow_index': f.flow_index, 'prediction': 'Malicious' if f.malicious else 'Safe',
           'attack_type': f.predicted_class if f.malicious else None, 'predicted_class': f.predicted_class, 'confidence': f.confidence,
           'confidence_level': f.confidence_level, 'severity': f.severity, 'prediction_set': f.prediction_set, 'ground_truth': f.ground_truth,
           'needs_review': f.confidence < rules.NEEDS_REVIEW_BELOW,
           'proto': f.flow.get('proto'), 'service': f.flow.get('service'), 'state': f.flow.get('state'), 'dur': f.flow.get('dur'),
           'sbytes': f.flow.get('sbytes'), 'dbytes': f.flow.get('dbytes'), 'spkts': f.flow.get('spkts'), 'dpkts': f.flow.get('dpkts')}
    if full:
        out['flow'] = f.flow
        out['has_incident_report'] = f.incident_report is not None
    return out


@router.get('/analyses/{analysis_id}/flows')
def list_flows(analysis_id: int, traffic: str | None = Query(None, pattern='^(safe|malicious)$'), attack_type: str | None = None,
               severity: str | None = None, confidence: str | None = Query(None, pattern='^(high|medium|low)$'), review: bool = False,
               uncertainty: str | None = Query(None, pattern='^(status|type|none)$'), flow_index: int | None = None,
               search: str | None = Query(None, max_length=64), page: int = Query(1, ge=1), page_size: int = Query(25, ge=5, le=100),
               db: Session = Depends(get_db), _: User = Depends(current_user)):
    get_analysis(db, analysis_id)
    q = select(FlowPrediction).where(FlowPrediction.analysis_id == analysis_id)
    if traffic:
        q = q.where(FlowPrediction.malicious == (traffic == 'malicious'))
    if attack_type:
        q = q.where(FlowPrediction.predicted_class == attack_type)
    if severity:
        q = q.where(FlowPrediction.severity == severity)
    if confidence:
        q = q.where(FlowPrediction.confidence_level == confidence)
    if review:
        q = q.where(FlowPrediction.confidence < rules.NEEDS_REVIEW_BELOW)
    if flow_index is not None:
        q = q.where(FlowPrediction.flow_index == flow_index)
    if uncertainty:
        members = cast(FlowPrediction.prediction_set, String)
        has_normal, several = members.like('%"Normal"%'), members.like('%,%')
        q = q.where({'status': has_normal & several,                                       # Normal and an attack type on the shortlist
                     'type': FlowPrediction.malicious & ~has_normal & several,             # attack, several attack types, no Normal
                     'none': ~several}[uncertainty])                                       # a single class on the shortlist
    if search:
        text = search.strip().lstrip('#')
        conditions = [func.lower(FlowPrediction.predicted_class).contains(text.lower())]
        if text.isdigit():
            conditions.append(FlowPrediction.flow_index == int(text))
        q = q.where(or_(*conditions))
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    items = db.scalars(q.order_by(FlowPrediction.flow_index).offset((page - 1) * page_size).limit(page_size)).all()
    return {'total': total, 'page': page, 'page_size': page_size, 'items': [flow_out(f) for f in items]}


def get_flow(db: Session, analysis_id: int, flow_id: int) -> FlowPrediction:
    flow = db.get(FlowPrediction, flow_id)
    if flow is None or flow.analysis_id != analysis_id:
        raise HTTPException(404, 'Flow not found')
    return flow


@router.get('/analyses/{analysis_id}/flows/{flow_id}')
def flow_detail(analysis_id: int, flow_id: int, request: Request, db: Session = Depends(get_db), _: User = Depends(current_user)):
    flow = get_flow(db, analysis_id, flow_id)
    detector = request.app.state.detector
    frame = pd.DataFrame([flow.flow])
    pred = detector.predict(frame)
    out = flow_out(flow, full=True)
    out['dataset_name'] = flow.analysis.dataset_name
    out['probabilities'] = dict(zip(CLASSES, pred.proba[0].round(6).tolist(), strict=True))
    out['explanation'] = detector.explain(frame, pred.predicted)[0]
    out['explanation_units'] = 'TreeSHAP contribution to the model score (log-odds) of the predicted class'
    out['descriptions'] = request.app.state.feature_descriptions
    return out


# ------------------------------------------------------------------------------------------------------------ 4. PDF security report
def report_out(r: AnalysisReport) -> dict:
    rag, record = r.rag or {}, r.record or {}
    return {'id': r.id, 'analysis_id': r.analysis_id, 'created_at': iso(r.created_at), 'pages': r.pages, 'size_bytes': r.size_bytes,
            'schema_version': r.schema_version or '1.0', 'has_record': bool(record),
            'components': record.get('components'), 'coverage': record.get('coverage'),
            'rag': {'llm_used': rag.get('llm_used', False), 'llm_model': rag.get('llm_model'), 'attack_types_covered': rag.get('attack_types_covered', 0),
                    'checked_items': rag.get('checked_items', 0), 'removed_items': rag.get('removed_items', 0)} if rag else None}


def priority_flows(db: Session, analysis_id: int, limit: int = 10) -> list[dict]:
    """Flows to look at first: heuristic severity (Critical first), then confidence. A review order, not confirmed threats."""
    order = {s: k for k, s in enumerate(rules.SEVERITY_ORDER)}
    rows = db.scalars(select(FlowPrediction).where(FlowPrediction.analysis_id == analysis_id, FlowPrediction.malicious)
                      .order_by(FlowPrediction.confidence.desc()).limit(2000)).all()
    rows = sorted(rows, key=lambda f: (order.get(f.severity, 9), -f.confidence, f.flow_index))[:limit]
    return [{'flow_id': f.id, 'flow_index': f.flow_index, 'class': f.predicted_class, 'severity_heuristic': f.severity, 'confidence': f.confidence,
             'prediction_set': f.prediction_set, 'proto': str(f.flow.get('proto')), 'service': str(f.flow.get('service')),
             'sbytes': f.flow.get('sbytes', 0), 'dbytes': f.flow.get('dbytes', 0)} for f in rows]


def write_pdf_report(app, analysis_id: int, user_id: int, progress) -> dict:
    """Build the versioned report record (schema 2), render the PDF from it, store both. Web, PDF and exports use the same record."""
    with app.state.sessions() as db:
        item = db.get(Analysis, analysis_id)
        if item is None:
            raise UserFacingError('The analysis no longer exists.')
        ensure_status_confidence(app, db, item)
        record = RecordBuilder(app, db, item, progress).build()
        record['detections']['priority_flows'] = priority_flows(db, analysis_id)
        report = AnalysisReport(analysis_id=item.id, filename='pending', pages=0, size_bytes=0, record=record, schema_version=record['schema_version'])
        db.add(report)
        db.flush()
        record['report_id'] = report.id
        progress('pdf', 'rendering')
        t0 = time.time()
        try:
            pdf, pages = render_record_pdf(record)
            record['components']['pdf'] = {'status': 'available', 'detail': f'{pages} pages'}
        except Exception as error:                                       # noqa: BLE001  the record is still stored and shown in the app
            pdf, pages = None, 0
            record['components']['pdf'] = {'status': 'failed', 'detail': f'{type(error).__name__}: {error}'}
        record['timings']['pdf'] = round(time.time() - t0, 3)
        record['timings']['total_report'] = round(sum(v for k, v in record['timings'].items() if k in ('total_record', 'pdf')), 3)
        if pdf is not None:
            folder = app.state.data_dir / 'reports'
            folder.mkdir(parents=True, exist_ok=True)
            report.filename = f'security_report_analysis_{item.id}_report_{report.id}.pdf'
            (folder / report.filename).write_bytes(pdf)
            report.pages, report.size_bytes = pages, len(pdf)
        report.record = record
        flag_modified(report, 'record')                                  # the dict was changed in place after the first flush
        db.commit()
        audit(db, db.get(User, user_id), 'report_generated', f'analysis #{item.id}: report #{report.id}, schema {record["schema_version"]}, {pages} pages')
        comps = record['components']
        return {'report_id': report.id, 'pages': pages, 'schema_version': record['schema_version'],
                'components': {k: v['status'] for k, v in comps.items()}, 'llm_detail': comps['llm']['detail'], 'checks_detail': comps['checks']['detail']}


@router.post('/analyses/{analysis_id}/report', status_code=202)
def start_report(analysis_id: int, request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    get_analysis(db, analysis_id)
    runner = request.app.state.report_jobs
    job = runner.submit(f'report:{analysis_id}', lambda progress: write_pdf_report(request.app, analysis_id, user.id, progress))
    return runner.describe(job)


@router.get('/analyses/{analysis_id}/report/job')
def report_job(analysis_id: int, request: Request, _: User = Depends(current_user)):
    runner = request.app.state.report_jobs
    job = runner.for_key(f'report:{analysis_id}')
    if job is None:
        raise HTTPException(404, 'No report job for this analysis')
    return runner.describe(job)


@router.get('/analyses/{analysis_id}/threat-intel')
def threat_intel(analysis_id: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    """Schema-1 reports only: their stored RAG threat intelligence. Schema-2 reports keep it inside the report record."""
    item = get_analysis(db, analysis_id)
    reports = [r for r in item.reports if r.rag]
    if not reports:
        raise HTTPException(404, 'No schema-1 threat intelligence for this analysis.')
    latest = max(reports, key=lambda r: r.id)
    return {'report_id': latest.id, 'created_at': iso(latest.created_at), **latest.rag}


@router.get('/reports')
def list_reports(db: Session = Depends(get_db), _: User = Depends(current_user)):
    """Every analysis with its latest report (if any): the Reports page."""
    return [analysis_out(a, full=False) for a in db.scalars(select(Analysis).order_by(Analysis.id.desc()))]


def get_report(db: Session, report_id: int) -> AnalysisReport:
    report = db.get(AnalysisReport, report_id)
    if report is None:
        raise HTTPException(404, 'Report not found')
    return report


@router.get('/reports/{report_id}/record')
def report_record(report_id: int, download: bool = False, db: Session = Depends(get_db), _: User = Depends(current_user)):
    """The versioned report record (schema 2) the web view, PDF and exports are rendered from. ?download=true saves it as JSON."""
    report = get_report(db, report_id)
    if not report.record:
        raise HTTPException(404, 'This is a schema-1 report (created before report records existed). Open its PDF or generate a new report.')
    record = {**report.record, 'views': report_views(report.record)}    # derived on read: same facts, current presentation for every 2.x record
    if download:
        name = f'security_report_analysis_{report.analysis_id}_report_{report.id}.json'
        return Response(json.dumps(record, indent=1, default=str), media_type='application/json',
                        headers={'Content-Disposition': f'attachment; filename="{name}"'})
    return record


@router.get('/reports/{report_id}/pdf')
def report_pdf(report_id: int, request: Request, download: bool = False, view: str = Query('detailed', pattern='^(detailed|summary)$'),
               db: Session = Depends(get_db), _: User = Depends(current_user)):
    """view=detailed: the PDF stored when the report was generated. view=summary: rendered from the same stored record on request (no model,
    retrieval or LLM call), so both PDFs show the same facts."""
    report = get_report(db, report_id)
    stem = f'security_report_{report.analysis.dataset_name.rsplit(".", 1)[0]}_analysis_{report.analysis_id}_report_{report.id}'
    if view == 'summary':
        if not report.record:
            raise HTTPException(404, 'This is a schema-1 report: only its detailed PDF exists. Generate the report again for a summary.')
        pdf, _pages = render_summary_pdf(report.record)
        kind = 'attachment' if download else 'inline'
        return Response(pdf, media_type='application/pdf', headers={'Content-Disposition': f'{kind}; filename="{stem}_summary.pdf"'})
    path = request.app.state.data_dir / 'reports' / report.filename
    if not path.exists():
        raise HTTPException(404, 'The PDF of this report is not available. Generate the report again.')
    return FileResponse(path, media_type='application/pdf', filename=f'{stem}_detailed.pdf', content_disposition_type='attachment' if download else 'inline')


class ActionUpdate(BaseModel):
    status: str = Field(pattern='^(open|reviewed|planned|completed_by_analyst|not_applicable)$')
    note: str = Field('', max_length=2000)


def action_states(db: Session, report_id: int) -> dict:
    rows = db.execute(select(ActionReview, User.email).join(User, ActionReview.user_id == User.id)
                      .where(ActionReview.report_id == report_id).order_by(ActionReview.id)).all()
    out: dict[str, dict] = {}
    for r, email in rows:
        entry = {'status': r.status, 'note': r.note, 'by': email, 'at': iso(r.created_at)}
        state = out.setdefault(r.action_id, {'history': []})
        state['history'].append(entry)
        state.update({k: entry[k] for k in ('status', 'note', 'by', 'at')})
    return out


@router.get('/reports/{report_id}/actions')
def list_action_states(report_id: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    """Analyst checklist of a report: latest state and full history per action (the app never executes actions)."""
    get_report(db, report_id)
    return action_states(db, report_id)


@router.post('/reports/{report_id}/actions/{action_id}')
def update_action_state(report_id: int, action_id: str, body: ActionUpdate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    report = get_report(db, report_id)
    known = {a['id'] for a in ((report.record or {}).get('actions') or {}).get('rows', [])}
    if action_id not in known:
        raise HTTPException(404, 'Unknown action for this report')
    db.add(ActionReview(report_id=report_id, action_id=action_id, status=body.status, note=body.note.strip(), user_id=user.id))
    db.commit()
    audit(db, user, 'action_review', f'report #{report_id} {action_id}: {body.status}')
    return action_states(db, report_id).get(action_id)


# ------------------------------------------------------------------------------------------------------------ 5. flow incident explanation (LLM)
def write_incident_report(app, flow_id: int, user_id: int, progress=None) -> dict:
    detector, engine = app.state.detector, app.state.reports
    with app.state.sessions() as db:
        flow = db.get(FlowPrediction, flow_id)
        if flow is None:
            raise UserFacingError('The flow no longer exists.')
        frame = pd.DataFrame([flow.flow])
        pred = detector.predict(frame)
        explanation = detector.explain(frame, pred.predicted)[0]
        result = engine.generate(flow.predicted_class, flow.confidence, flow.prediction_set, explanation, progress=progress).to_dict()
        if progress:
            progress('saving')
        if flow.incident_report is not None:
            db.delete(flow.incident_report)
            db.flush()
        db.add(FlowReport(flow_id=flow.id, content=result, llm_used=result['llm_used']))
        db.commit()
        audit(db, db.get(User, user_id), 'incident_report', f'flow {flow_id} (LLM used: {result["llm_used"]})')
    return {'flow_id': flow_id}


@router.post('/analyses/{analysis_id}/flows/{flow_id}/incident-report', status_code=202)
def start_incident_report(analysis_id: int, flow_id: int, request: Request, response: Response, wait: bool = False,
                          db: Session = Depends(get_db), user: User = Depends(current_user)):
    get_flow(db, analysis_id, flow_id)
    if wait:
        write_incident_report(request.app, flow_id, user.id)
        response.status_code = 200
        db.expire_all()
        return get_flow(db, analysis_id, flow_id).incident_report.content
    runner = request.app.state.incident_jobs
    job = runner.submit(f'incident:{flow_id}', lambda progress: write_incident_report(request.app, flow_id, user.id, progress))
    return runner.describe(job)


@router.get('/analyses/{analysis_id}/flows/{flow_id}/incident-report/job')
def incident_job(analysis_id: int, flow_id: int, request: Request, _: User = Depends(current_user)):
    runner = request.app.state.incident_jobs
    job = runner.for_key(f'incident:{flow_id}')
    if job is None:
        raise HTTPException(404, 'No incident report job for this flow')
    return runner.describe(job)


@router.get('/analyses/{analysis_id}/flows/{flow_id}/incident-report')
def incident_report(analysis_id: int, flow_id: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    flow = get_flow(db, analysis_id, flow_id)
    if flow.incident_report is None:
        raise HTTPException(404, 'No incident report yet')
    return flow.incident_report.content


# ------------------------------------------------------------------------------------------------------------ overview + model
@router.get('/overview')
def overview(request: Request, db: Session = Depends(get_db), _: User = Depends(current_user)):
    """Statistics over ALL analyses (the Overview page). Per-dataset numbers belong to the Results page."""
    items = db.scalars(select(Analysis).order_by(Analysis.id.desc())).all()
    total = sum(a.total_flows for a in items)
    malicious = sum(a.malicious_count for a in items)
    by_type: dict[str, int] = {}
    for a in items:
        for d in a.summary.get('attack_distribution', []):
            by_type[d['attack_type']] = by_type.get(d['attack_type'], 0) + d['count']
    evaluation = request.app.state.model_card.get('evaluation') or {}
    return {'files_analyzed': len(items), 'total_flows': total, 'total_threats': malicious, 'safe_flows': total - malicious,
            'model_accuracy': evaluation.get('accuracy'), 'model_binary_accuracy': (evaluation.get('binary') or {}).get('accuracy'),
            'model_attack_f1': (evaluation.get('binary') or {}).get('f1'),
            'attacks_by_type': [{'attack_type': k, 'count': v, 'severity': rules.SEVERITY[k]} for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])],
            'risk_levels': {lv: sum(1 for a in items if a.risk_level == lv) for lv in ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')},
            'recent': [analysis_out(a, full=False) for a in items[:6]]}


@router.get('/model')
def model_info(request: Request, _: User = Depends(current_user)):
    d = request.app.state.detector
    return {'model_card': request.app.state.model_card, 'rules': rules.rules(),
            'grouped_evaluation': rules.grouped_evaluation(request.app.state.model_card.get('evaluation')),
            'calibration': {'temperature': d.temperature, 'alpha': d.alpha,
                            'class_thresholds': dict(zip(CLASSES, d.thresholds.round(6).tolist(), strict=True))},
            'bundle': {k: d.manifest.get(k) for k in ('created', 'source_commit', 'detector', 'verification')},
            'features': [{'name': c, 'description': request.app.state.feature_descriptions.get(c, '')} for c in d.predictors]}


@router.post('/predict')
def predict(request: Request, body: PredictRequest, _: User = Depends(current_user)):
    """Stateless prediction for integrations: nothing is stored."""
    detector = request.app.state.detector
    try:
        frame = pd.DataFrame(body.flows)
        pred = detector.predict(frame)
    except InputError as error:
        raise HTTPException(422, str(error)) from error
    return [{'predicted_class': CLASSES[pred.predicted[i]], 'confidence': float(pred.proba[i, pred.predicted[i]]),
             'prediction_set': [CLASSES[j] for j in np.flatnonzero(pred.sets[i])],
             'probabilities': dict(zip(CLASSES, pred.proba[i].round(6).tolist(), strict=True))} for i in range(len(frame))]
