// Typed client for the FastAPI backend. The token is kept in localStorage; a 401 logs the user out.

export const CLASSES = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal', 'Reconnaissance', 'Shellcode', 'Worms'] as const
export type FlowClass = (typeof CLASSES)[number]
export type Severity = 'Critical' | 'High' | 'Medium' | 'Low' | 'None'
export type ConfidenceLevel = 'high' | 'medium' | 'low'
export type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'

export interface User { id: number; email: string; name: string; role: string }

// ---------------------------------------------------------------------------------------------- upload + validation
export interface ValidationError { title: string; message: string; items: { name: string; description: string }[] }
export interface UploadReport {
  upload_id: string | null; filename: string; size_bytes: number; rows: number | null; columns: number | null
  valid: boolean; errors: ValidationError[]; warnings: string[]; has_labels: boolean; required_features: number
}

// ---------------------------------------------------------------------------------------------- jobs
export interface Job<R = Record<string, number>> {
  job_id: string; key: string; status: 'queued' | 'running' | 'done' | 'failed'; stage: string; detail: string; stages: string[]
  queue_position: number; error: string; result: R; elapsed: number
}

// ---------------------------------------------------------------------------------------------- the analysis object
export interface AttackShare {
  attack_type: FlowClass; count: number; percentage_of_malicious: number; percentage_of_all: number
  average_confidence: number; severity: Severity; needs_review: number
}
export interface ClassResult {
  class: FlowClass; flows: number; correct: number; in_set: number; flagged_as_attack: number
  confused_with: FlowClass | null; confused_count: number
}
/** Only when the uploaded file itself carried true labels. */
export interface Evaluation {
  labelled_flows: number; attacks: number; attacks_caught: number; normal: number; normal_flagged: number
  attack_vs_normal_correct: number; exact_class_correct: number; truth_in_set: number; per_class: ClassResult[]
}
export interface Rules {
  confidence_levels: Record<ConfidenceLevel, string>; needs_review: string; attack_type_uncertain: string
  severity: Record<string, Severity>; severity_reason: Record<string, string>; risk: { level: RiskLevel; rule: string }[]; confidence_meaning: string
}
export interface AnalysisData {
  total_flows: number; safe_count: number; malicious_count: number; safe_percentage: number; malicious_percentage: number
  average_confidence: number; confidence_distribution: Record<ConfidenceLevel, number>
  needs_review_count: number; attack_type_uncertain_count: number
  attack_distribution: AttackShare[]; severity_distribution: Record<'Critical' | 'High' | 'Medium' | 'Low', number>
  risk: { level: RiskLevel; rule: string; explanation: string }
  summary: {
    most_detected_attack: { attack_type: FlowClass; count: number } | null; attack_types_detected: number
    highest_confidence_attack: { attack_type: FlowClass; average_confidence: number } | null; flows_requiring_review: number
  }
  rules: Rules; evaluation?: Evaluation
  /** How sure the model is that a flow is an attack (any type) or Normal: the attack-class probabilities added together. */
  status_confidence_distribution?: Record<ConfidenceLevel, number>; average_status_confidence?: number; status_needs_review_count?: number
}
export interface ReportResult { report_id: number; pages: number; llm_used?: boolean; attack_types_covered?: number; checked_items?: number; removed_items?: number }
export interface ReportInfo {
  id: number; analysis_id: number; created_at: string; pages: number; size_bytes: number
  /** '2.0' = versioned report record (web, PDF, JSON and Markdown come from it); '1.0' = older PDF-only report. */
  schema_version: string; has_record: boolean; components: Record<string, ComponentStatus> | null; coverage: ReportRecord['coverage'] | null
  rag: { llm_used: boolean; llm_model: string | null; attack_types_covered: number; checked_items: number; removed_items: number } | null
}
export interface AnalysisSummary {
  id: number; dataset_name: string; source: string; created_at: string; seconds: number
  total_flows: number; malicious_count: number; safe_count: number; risk_level: RiskLevel; report: ReportInfo | null
  malicious_percentage?: number; average_confidence?: number; has_labels?: boolean
}
export interface AnalysisResult extends AnalysisSummary { analysis: AnalysisData }

// ---------------------------------------------------------------------------------------------- flows
export interface FlowRow {
  id: number; analysis_id: number; flow_index: number; prediction: 'Safe' | 'Malicious'; attack_type: FlowClass | null
  predicted_class: FlowClass; confidence: number; confidence_level: ConfidenceLevel; severity: Severity
  prediction_set: FlowClass[]; ground_truth: string | null; needs_review: boolean
  proto: string; service: string; state: string; dur: number; sbytes: number; dbytes: number; spkts: number; dpkts: number
}
export interface Explanation { feature: string; value: string | number | null; contribution: number }
export interface FlowDetail extends FlowRow {
  flow: Record<string, string | number>; has_incident_report: boolean; dataset_name: string
  probabilities: Record<FlowClass, number>; explanation: Explanation[]; explanation_units: string; descriptions: Record<string, string>
}
export interface Paged<T> { total: number; page: number; page_size: number; items: T[] }
export interface FlowFilters {
  traffic?: string; attack_type?: string; severity?: string; confidence?: string; review?: boolean; search?: string; page?: number; page_size?: number
  /** status = Normal and an attack type both on the 90% shortlist; type = attack, several attack types; none = single-class shortlist */
  uncertainty?: 'status' | 'type' | 'none'; flow_index?: number
}

// ---------------------------------------------------------------------------------------------- incident explanation (LLM)
export interface ReportItem { text?: string; id?: string; name?: string | null; cites: string[]; reason?: string }
export interface RemovedItem { part: string; text: string; cites: string[]; reason: string }
export interface IncidentReport {
  alert_class: string; summary: string; facts: Record<string, string>
  sources: { id: string; source: string; location: string }[]
  findings: ReportItem[]; claims: ReportItem[]; techniques: ReportItem[]; actions: ReportItem[]
  possible_attacks?: { class: string; predicted: boolean; mapping_confidence: string; techniques: { id: string; name: string | null }[] }[]
  evidence_explanation?: ReportItem[]; checked_items?: number; removed?: RemovedItem[]
  removed_items: number; llm_used: boolean; llm_model: string | null; seconds: number; note: string
}
/** Dataset-level threat intelligence (RAG + LLM, verified), created with the PDF report. */
export interface ThreatSection extends IncidentReport {
  attack_type: FlowClass; count: number; severity: Severity; percentage_of_malicious: number; average_confidence: number
  representative_flow: { id: number; flow_index: number; confidence: number }
}
export interface ThreatIntel {
  report_id: number; created_at: string; method: string; llm_model: string; llm_used: boolean; llm_online: boolean
  attack_types_covered: number; attack_types_total: number; checked_items: number; removed_items: number; seconds: number
  sections: ThreatSection[]; note: string
}

// ---------------------------------------------------------------------------------------------- security report record (schema 2)
/** Where a statement comes from. Shown next to every statement in the web view, PDF and exports. */
export type Origin = 'observed' | 'batch_statistic' | 'model_output' | 'xai_attribution' | 'retrieved_guidance' | 'llm_synthesis' | 'analyst_proposed' | 'heuristic'
export interface ComponentStatus { status: string; detail?: string; reasons?: string[]; method?: string }
export interface Support { source: string; start: number; end: number; span: string; overlap: number; method: string }
export interface Statement {
  id: string; text: string; technique: string | null; cites: string[]; origin: Origin; status: string; check: string; round: number; support: Support | null
}
export interface RejectedStatement { part: string; text: string; cites: string[]; reason: string; round?: number }
export interface Contribution { feature: string; value: string; contribution: number; name?: string; direction?: 'supports' | 'opposes'; definition?: string; reference?: string | null }
export interface FlowCard {
  id: string; flow_id: number; flow_index: number; selection: string
  prediction: {
    class: FlowClass; malicious: boolean; confidence: number; confidence_level: ConfidenceLevel; severity_heuristic: Severity
    prediction_set: FlowClass[]; normal_in_set: boolean; probabilities: Record<FlowClass, number>; origin: Origin
  }
  values: { feature: string; name: string; value: string; definition: string; reference: string | null }[]
  xai: {
    status: string; method: string; class: FlowClass; units: string; base_value: number; margin: number; temperature: number; class_offset: number
    deployed_class_score: number; remainder_of_other_features: number; other_features: number; top: Contribution[]; all: Contribution[]
    one_sided: boolean; how_to_read: string
  }
  lime: { status: string; reason: string }
  facts: Record<string, string>
  /** schema 2.1: short code-written reading of the chart, other classes not ruled out, what would confirm it */
  interpretation?: string | null; alternatives?: { class: FlowClass; probability: number; in_set: boolean }[]; missing_confirmation?: string | null
  /** Up to five reasoning points (how sure, main reasons, how typical, why not the runner-up, in this file); AI-written ones passed the checks. */
  reasoning?: { slot: string; title: string; text: string; cites: string[]; origin: Origin }[]
  status_probability?: number
  llm: {
    status: string; note: string; model: string | null; seconds: number; facts: Record<string, string>
    retrieval: { status: string; passages: number; duplicates_removed: number; guidance_rejected_low_score: number; min_score: number } | null
    findings: { text: string; cites: string[]; origin: Origin; status: string }[]
    explanation: Statement[]; techniques: Statement[]; context: Statement[]; actions: Statement[]
    mapped_techniques: { class: FlowClass; predicted: boolean; mapping_confidence: string; techniques: { id: string; name: string | null }[]; origin: Origin; note: string }[]
    checked: number; removed: RejectedStatement[]
  } | null
  ground_truth_note: string
}
export interface SourcePassage {
  id: string; chunk_id: string | null; source: string; publisher: string | null; title: string | null; version: string | null; location: string
  url: string | null; document_url: string | null; retrieved: string | null; pdf_created: string | null; document_sha256: string | null
  content_sha256: string; text: string; excerpt?: string | null
  retrieval: { method: string | null; score: number | null; query: string | null }; used_by: string[]; trust: string
}
export interface ActionRow {
  id: string; phase: 'Verify' | 'Contain if confirmed' | 'Recover'; scope: string; text: string; reason: string; cites: string[]; origin: Origin
  check?: string | null; support?: Support | null; applicability: string; owner: string; disruption: string; validation: string
  kind?: string; title?: string; rollback?: string | null; priority?: number
}
export interface GroupFinding {
  class: FlowClass; status: string; population: number; analysed?: number; origin?: Origin; method?: string; reason?: string
  top_supporting?: { feature: string; name: string; plain: string; flows: number; share: number; value_summary: string; reference: string | null }[]
  most_frequent_opposing?: { feature: string; name: string; flows: number; share: number }[]
}
export interface OverviewBullet { text: string; origin: Origin; cites: string[]; kind: string; scope?: string }
/** The Summary view: computed on the server from the same record (app.core.report_views), never a separate analysis. */
export interface SummaryViewData {
  overview: { bullets: OverviewBullet[]; words: number; method: string }
  examples: { class: FlowClass; flow_id: string; flow_index: number; selection: string; model: string | null; origin: Origin; note: string
    points: { title?: string; text: string; exact: string; cites: string[]; origin?: Origin }[] }[]
  detected: { question: string; answer: string; groups: { class: FlowClass; count: number; share: number; average_confidence: number; meaning: string; meaning_source: string
    context: { text: string; cites: string[]; origin: Origin; note: string } | null }[] }
  actions: { id: string; phase: ActionRow['phase']; what: string; how: string; when: string; check: string; owner: string; cites: string[] }[]
  outcome: { evidence_needed: string; confirms: string; weakens: string[]; mitigation_success: string; origin: Origin }
  limits: string[]
  sources: { id: string; source: string; location: string; url: string | null }[]
}
export interface ReportDocument {
  source: string; publisher: string | null; title: string | null; version: string | null; download_url: string | null
  document_sha256: string | null; retrieved: string | null; pdf_created: string | null; passages: number
}
export interface ReportRecord {
  schema: string; schema_version: string; template_version: string; generated_at: string; question: string; report_id: number
  provenance: {
    analysis: { id: number; dataset_name: string; created_at: string; flows: number; input_sha256: string | null; input_note: string | null }
    detector: { name: string; version: string; bundle_created: string | null; bundle_source_commit: string | null; files_sha256: Record<string, string> }
    calibration: { temperature: number; alpha: number; class_offsets: number[] }
    explanation: { method: string; exactness: string }
    knowledge: { downloaded: string | null; chunks_sha256: string | null }
    llm: { model: string; runtime: string; settings: Record<string, number>; note: string }
    application: { code_commit: string | null; report_schema: string; template: string }
  }
  components: Record<'detection' | 'xai' | 'retrieval' | 'llm' | 'checks' | 'pdf', ComponentStatus> & { group_xai?: ComponentStatus }
  coverage: { flows_explained: number; flows_total: number; attack_types_with_llm: number; attack_types_detected: number; selection_rule?: string; group_flows_explained?: number }
  summary: {
    scope: string; predicted_normal: number; predicted_attack: number; predicted_normal_share: number; predicted_attack_share: number
    average_confidence: number; needs_review: number; attack_status_uncertain: number; attack_type_uncertain: number
    triage_priority: { level: RiskLevel; rule: string; origin: Origin; meaning: string }
    most_important_uncertainty: string; next_check: string
    status_confidence?: { levels: Record<ConfidenceLevel, number>; average: number; meaning: string } | null
  }
  detections: {
    total: number
    by_class: { class: FlowClass; count: number; share: number; average_confidence: number; severity_heuristic: Severity; status_uncertain: number; type_uncertain: number }[]
    confidence_levels: Record<ConfidenceLevel, number>
    sets: { singleton: number; multiple: number; contains_normal: number; attack_status_uncertain: number; attack_type_uncertain: number; predicted_attack_with_normal_in_set: number; predicted_normal_with_attack_in_set: number }
    notes: string[]; shift_evidence: string[]
    priority_flows: { flow_id: number; flow_index: number; class: FlowClass; severity_heuristic: Severity; confidence: number; prediction_set: FlowClass[]; proto: string; service: string; sbytes: number; dbytes: number }[]
  }
  flows: FlowCard[]
  group_findings?: GroupFinding[]
  overview?: SummaryViewData['overview']
  views: { summary: SummaryViewData }
  sources: { register: SourcePassage[]; documents?: ReportDocument[]; note: string }
  actions: { rows: ActionRow[]; prevention: { scope: string; text: string; cites: string[]; origin: Origin; note: string }[]; note: string }
  limits: { unknown: string[]; human_review: string; origins: Record<Origin, string> }
  appendix: { model_card: ModelInfo['model_card']; note: string }
  evaluation: (Evaluation & { scope: string }) | null
  timings: Record<string, number>
}
export type ReportView = 'summary' | 'detailed'
export type ActionStatus = 'open' | 'reviewed' | 'planned' | 'completed_by_analyst' | 'not_applicable'
export interface ActionEntry { status: ActionStatus; note: string; by: string; at: string }
export interface ActionState extends ActionEntry { history: ActionEntry[] }

// ---------------------------------------------------------------------------------------------- overview + model
export interface Overview {
  files_analyzed: number; total_flows: number; total_threats: number; safe_flows: number
  model_accuracy: number | null; model_binary_accuracy: number | null; model_attack_f1: number | null
  attacks_by_type: { attack_type: FlowClass; count: number; severity: Severity }[]
  risk_levels: Record<RiskLevel, number>; recent: AnalysisSummary[]
}
export interface Metrics { precision: number; recall: number; f1: number }
export interface ModelInfo {
  model_card: {
    name: string; version: string; task?: string; limitations: string[]; classes?: FlowClass[]; input_features?: number; model_columns?: number
    dataset?: { name: string; training_rows: number; test_rows: number; fitting_rows: number; model_selection_rows: number; calibration_rows: number }
    evaluation?: {
      split: string; rows: number; accuracy: number; macro: Metrics; weighted: Metrics
      binary: Metrics & { accuracy: number; false_positive_rate: number }
      per_class: (Metrics & { class: FlowClass; support: number; ceiling_f1?: number })[]
      confusion_matrix: { labels: FlowClass[]; matrix: number[][] }
      top2_accuracy?: number
      ceiling?: { accuracy: number; macro_f1: number; attack_f1: number; meaning: string }
    }
    in_distribution?: { split: string; accuracy: number; macro_f1: number; attack_f1: number }
    comparison?: { split: string; source: string; rows: { model: string; runs: number; macro_f1: number; accuracy: number; attack_recall: number; false_positive_rate: number; deployed: boolean }[] }
  }
  rules: Rules
  /** DoS, Analysis and Backdoor counted as one group, computed from the official-test confusion matrix (not a new evaluation). */
  grouped_evaluation: GroupedEvaluation | null
  calibration: { temperature: number; alpha: number; class_thresholds: Record<string, number> }
  bundle: { created?: string; source_commit?: string; detector?: string; verification?: Record<string, { rows: number; max_abs_difference: number; same_predicted_class: boolean }> }
  features: { name: string; description: string }[]
}
export interface GroupedEvaluation {
  groups: string[]; per_class: (Metrics & { class: string; support: number })[]; accuracy: number; macro_f1: number; group: string
  within_group_confusions: number; note: string
}
export interface Health { status: string; model: string | null; model_loaded: boolean; report_generator: boolean; llm_available: boolean }

// ---------------------------------------------------------------------------------------------- transport
const TOKEN_KEY = 'nad.token'
export const tokenStore = {
  get: () => { try { return localStorage.getItem(TOKEN_KEY) } catch { return null } },
  set: (t: string) => { try { localStorage.setItem(TOKEN_KEY, t) } catch { /* storage unavailable */ } },
  clear: () => { try { localStorage.removeItem(TOKEN_KEY) } catch { /* storage unavailable */ } },
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) { super(message); this.status = status }
}

let onUnauthorized: () => void = () => {}
export const setUnauthorizedHandler = (fn: () => void) => { onUnauthorized = fn }

const OFFLINE = 'Cannot reach the server. Check that the backend window is still open (start-app.cmd), then try again.'

function errorMessage(status: number, statusText: string, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  if (detail) return 'The request was not accepted. Please check the input and try again.'
  if (status === 502 || status === 503 || status === 504) return OFFLINE
  if (status >= 500) return `The server hit an error (${status}). Please try again; details are in the server log.`
  return statusText || `Request failed (${status})`
}

async function send(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  const token = tokenStore.get()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !(init.body instanceof FormData) && !(init.body instanceof URLSearchParams)) headers.set('Content-Type', 'application/json')
  let res: Response
  try {
    res = await fetch(`/api${path}`, { ...init, headers })
  } catch {
    throw new ApiError(0, OFFLINE)
  }
  if (res.status === 401 && path !== '/auth/login') { tokenStore.clear(); onUnauthorized() }
  if (!res.ok) {
    let body: unknown = null
    try { body = await res.json() } catch { /* not JSON */ }
    throw new ApiError(res.status, errorMessage(res.status, res.statusText, body))
  }
  return res
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await send(path, init)
  return (res.status === 204 ? null : res.json()) as Promise<T>
}

/** Upload with progress (fetch cannot report upload progress). */
function uploadWithProgress<T>(path: string, form: FormData, onProgress: (fraction: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `/api${path}`)
    const token = tokenStore.get()
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`)
    xhr.upload.onprogress = e => { if (e.lengthComputable) onProgress(e.loaded / e.total) }
    xhr.upload.onload = () => onProgress(1)
    xhr.onerror = () => reject(new ApiError(0, OFFLINE))
    xhr.onload = () => {
      let body: unknown = null
      try { body = JSON.parse(xhr.responseText) } catch { /* not JSON */ }
      if (xhr.status === 401) { tokenStore.clear(); onUnauthorized() }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T)
      else reject(new ApiError(xhr.status, errorMessage(xhr.status, xhr.statusText, body)))
    }
    xhr.send(form)
  })
}

/** Fetch a file with the token and either save it or return an object URL (for the PDF preview). */
export async function fetchBlob(path: string): Promise<{ url: string; name: string }> {
  const res = await send(path)
  const name = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/.exec(res.headers.get('Content-Disposition') ?? '')?.[1] ?? 'download'
  return { url: URL.createObjectURL(await res.blob()), name: decodeURIComponent(name) }
}

export async function downloadFile(path: string) {
  const { url, name } = await fetchBlob(path)
  const a = document.createElement('a')
  a.href = url; a.download = name; a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

const q = (params: Record<string, string | number | boolean | undefined | null>) => {
  const s = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== '' && v !== false) s.set(k, String(v)) })
  const text = s.toString()
  return text ? `?${text}` : ''
}

export const api = {
  health: () => request<Health>('/health'),
  login: (email: string, password: string) =>
    request<{ access_token: string; user: User }>('/auth/login', { method: 'POST', body: new URLSearchParams({ username: email, password }) }),
  me: () => request<User>('/auth/me'),

  upload: (file: File, onProgress: (fraction: number) => void = () => {}) => {
    const f = new FormData(); f.append('file', file)
    return uploadWithProgress<UploadReport>('/uploads', f, onProgress)
  },
  uploadSample: (perClass = 30) => request<UploadReport>(`/uploads/sample${q({ per_class: perClass })}`, { method: 'POST' }),
  discardUpload: (id: string) => request<null>(`/uploads/${id}`, { method: 'DELETE' }),
  downloadSample: (perClass = 30) => downloadFile(`/samples/flows.csv${q({ per_class: perClass })}`),
  startAnalysis: (uploadId: string) => request<Job<{ analysis_id: number }>>('/analyses', { method: 'POST', body: JSON.stringify({ upload_id: uploadId }) }),
  analysisJob: (jobId: string) => request<Job<{ analysis_id: number }>>(`/analyses/jobs/${jobId}`),

  analyses: () => request<AnalysisSummary[]>('/analyses'),
  analysis: (id: number) => request<AnalysisResult>(`/analyses/${id}`),
  deleteAnalysis: (id: number) => request<null>(`/analyses/${id}`, { method: 'DELETE' }),
  flows: (id: number, f: FlowFilters) => request<Paged<FlowRow>>(`/analyses/${id}/flows${q({ ...f })}`),
  flow: (id: number, flowId: number) => request<FlowDetail>(`/analyses/${id}/flows/${flowId}`),

  threatIntel: (id: number) => request<ThreatIntel>(`/analyses/${id}/threat-intel`),
  startReport: (id: number) => request<Job<ReportResult>>(`/analyses/${id}/report`, { method: 'POST' }),
  reportJob: (id: number) => request<Job<ReportResult>>(`/analyses/${id}/report/job`),
  reports: () => request<AnalysisSummary[]>('/reports'),
  reportPdfPath: (reportId: number, download = false, view: ReportView = 'detailed') => `/reports/${reportId}/pdf${q({ download, view: view === 'detailed' ? undefined : view })}`,
  reportRecord: (reportId: number) => request<ReportRecord>(`/reports/${reportId}/record`),
  reportRecordPath: (reportId: number) => `/reports/${reportId}/record?download=true`,
  actionStates: (reportId: number) => request<Record<string, ActionState>>(`/reports/${reportId}/actions`),
  updateAction: (reportId: number, actionId: string, status: ActionStatus, note = '') =>
    request<ActionState>(`/reports/${reportId}/actions/${encodeURIComponent(actionId)}`, { method: 'POST', body: JSON.stringify({ status, note }) }),

  startIncident: (id: number, flowId: number) => request<Job>(`/analyses/${id}/flows/${flowId}/incident-report`, { method: 'POST' }),
  incidentJob: (id: number, flowId: number) => request<Job>(`/analyses/${id}/flows/${flowId}/incident-report/job`),
  incident: (id: number, flowId: number) => request<IncidentReport>(`/analyses/${id}/flows/${flowId}/incident-report`),

  overview: () => request<Overview>('/overview'),
  model: () => request<ModelInfo>('/model'),
}
