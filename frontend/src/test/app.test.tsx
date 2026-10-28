import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import Accuracy from '../components/Accuracy'
import { ConfidenceLevels, TrafficDonut } from '../components/charts'
import ReportView from '../components/ReportView'
import { PredictionBadge, RiskBadge, SeverityBadge, StepList } from '../components/ui'
import type { AnalysisData, Evaluation, IncidentReport } from '../lib/api'
import { bytes, pct, sortedEntries, when } from '../lib/format'
import { reportToMarkdown } from '../lib/report'

const rules: AnalysisData['rules'] = {
  confidence_levels: { high: '>= 90%', medium: '70% to 89.9%', low: '< 70%' }, needs_review: 'confidence below 70%',
  attack_type_uncertain: 'x', severity: { DoS: 'High' }, severity_reason: {}, risk: [], confidence_meaning: 'x',
}

describe('formatting', () => {
  it('formats percentages, sizes and sorts counts', () => {
    expect(pct(0.9818)).toBe('98.2%')
    expect(pct(undefined)).toBe('–')
    expect(bytes(2048)).toBe('2.0 KB')
    expect(sortedEntries({ a: 1, b: 5 })[0][0]).toBe('b')
  })
  it('reads server times without a zone as UTC', () => {
    expect(when('2026-10-18T16:42:13')).toBe(new Date('2026-10-18T16:42:13Z').toLocaleString())
    expect(when('2026-10-18T16:42:13+00:00')).toBe(new Date('2026-10-18T16:42:13Z').toLocaleString())
  })
})

describe('badges and steps', () => {
  it('shows prediction, severity and risk', () => {
    render(<><PredictionBadge malicious /><PredictionBadge malicious={false} /><SeverityBadge severity="Critical" /><RiskBadge level="HIGH" /></>)
    expect(screen.getByText('Predicted attack')).toBeInTheDocument()
    expect(screen.getByText('Predicted Normal')).toBeInTheDocument()
    expect(screen.getByText('Critical')).toBeInTheDocument()
    expect(screen.getByText('HIGH')).toBeInTheDocument()
  })
  it('marks completed, processing, pending and failed stages', () => {
    const steps = [{ key: 'a', label: 'A' }, { key: 'b', label: 'B' }, { key: 'c', label: 'C' }]
    const { container, rerender } = render(<StepList steps={steps} current={1} />)
    expect([...container.querySelectorAll('li')].map(li => li.dataset.state)).toEqual(['completed', 'processing', 'pending'])
    rerender(<StepList steps={steps} current={1} failed />)
    expect(container.querySelectorAll('li')[1].dataset.state).toBe('failed')
    rerender(<StepList steps={steps} current={0} done />)
    expect([...container.querySelectorAll('li')].every(li => li.dataset.state === 'completed')).toBe(true)
  })
})

describe('charts', () => {
  it('traffic donut shows counts and shares', () => {
    render(<TrafficDonut safe={17} malicious={283} />)
    expect(screen.getByText('283')).toBeInTheDocument()
    expect(screen.getAllByText('94.3% of 300 flows').length).toBe(1)
    expect(screen.getByText('5.7% of 300 flows')).toBeInTheDocument()
  })
  it('confidence levels use the real thresholds and counts', () => {
    const a = { total_flows: 300, confidence_distribution: { high: 104, medium: 58, low: 138 }, rules } as unknown as AnalysisData
    render(<ConfidenceLevels a={a} />)
    expect(screen.getByText('(< 70%)')).toBeInTheDocument()
    expect(screen.getByText('138')).toBeInTheDocument()
    expect(screen.getByText('· 46.0%')).toBeInTheDocument()
  })
})

describe('labels check', () => {
  it('states the results in plain words with counts', () => {
    const evaluation: Evaluation = {
      labelled_flows: 300, attacks: 270, attacks_caught: 261, normal: 30, normal_flagged: 5, attack_vs_normal_correct: 286,
      exact_class_correct: 183, truth_in_set: 250,
      per_class: [{ class: 'DoS', flows: 30, correct: 9, in_set: 25, flagged_as_attack: 30, confused_with: 'Backdoor', confused_count: 12 }],
    }
    render(<Accuracy evaluation={evaluation} />)
    expect(screen.getByText('96.7%')).toBeInTheDocument()
    expect(screen.getByText('16.7%')).toBeInTheDocument()
    expect(screen.getByText('× 12')).toBeInTheDocument()
  })
})

const incident: IncidentReport = {
  alert_class: 'DoS', summary: 'The detector flags this flow as DoS (calibrated probability 0.32).',
  facts: { E1: 'Predicted class: DoS with calibrated probability 0.32' },
  sources: [{ id: 'S1', source: 'MITRE ATT&CK', location: 'T1498' }],
  findings: [{ text: 'The detector classifies this flow as DoS with calibrated probability 0.32.', cites: ['E1'] }],
  claims: [], techniques: [{ id: 'T1498', name: 'Network Denial of Service', cites: ['S1'] }],
  actions: [{ text: 'Filter Network Traffic: block the flood upstream.', cites: ['S1'] }],
  possible_attacks: [{ class: 'DoS', predicted: true, mapping_confidence: 'high', techniques: [{ id: 'T1498', name: 'Network Denial of Service' }] }],
  removed_items: 0, llm_used: true, llm_model: 'qwen2.5:7b', seconds: 11, note: '',
}

describe('flow incident explanation', () => {
  it('renders verified items with citations and ATT&CK links', () => {
    render(<ReportView alertId={7} report={incident} />)
    expect(screen.getByText(/Filter Network Traffic/)).toBeInTheDocument()
    expect(screen.getAllByRole('link', { name: /T1498/ })[0]).toHaveAttribute('href', 'https://attack.mitre.org/techniques/T1498/')
    expect(screen.getByText(/0 unsupported item\(s\) removed/)).toBeInTheDocument()
  })
  it('separates exact evidence, verified LLM text and removed items', () => {
    render(<ReportView alertId={7} report={{ ...incident, checked_items: 4, removed_items: 1,
      evidence_explanation: [{ text: 'The model is 0.32 confident that this flow is DoS.', cites: ['E1'], reason: 'supported' }],
      removed: [{ part: 'evidence', text: 'A large sbytes value', cites: ['E1'], reason: 'value judgement "large" not supported by the evidence' }] }} />)
    expect(screen.getByText('Why the model predicted DoS')).toBeInTheDocument()
    expect(screen.getByText('The model is 0.32 confident that this flow is DoS.')).toBeInTheDocument()
    expect(screen.getAllByText('LLM · verified').length).toBeGreaterThan(0)
    expect(screen.getAllByText('exact, from the model').length).toBeGreaterThan(0)
    expect(screen.getByText('Removed by the verifier')).toBeInTheDocument()
    expect(screen.getByText(/value judgement "large"/)).toBeInTheDocument()
  })
  it('exports Markdown with every section', () => {
    const md = reportToMarkdown(7, incident)
    expect(md).toContain('- T1498 Network Denial of Service [S1]')
    expect(md).toContain('## Recommended actions')
    expect(md).toContain('DoS (predicted): T1498 Network Denial of Service')
  })
})


describe('two confidences', () => {
  it('shows attack-or-normal confidence first and switches to the exact type', () => {
    const a = { total_flows: 300, confidence_distribution: { high: 92, medium: 57, low: 151 }, status_confidence_distribution: { high: 263, medium: 26, low: 11 }, rules } as unknown as AnalysisData
    render(<ConfidenceLevels a={a} />)
    expect(screen.getByRole('radio', { name: 'Attack or normal?' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByText('263')).toBeInTheDocument()
    expect(screen.getByText(/usually sure about this/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('radio', { name: 'Exact attack type?' }))
    expect(screen.getByText('151')).toBeInTheDocument()
    expect(screen.getByText(/some attack types \(for example DoS, Analysis and Backdoor\)/)).toBeInTheDocument()
  })
  it('falls back to the class confidence for older analyses', () => {
    const a = { total_flows: 300, confidence_distribution: { high: 92, medium: 57, low: 151 }, rules } as unknown as AnalysisData
    render(<ConfidenceLevels a={a} />)
    expect(screen.queryByRole('radio')).not.toBeInTheDocument()
    expect(screen.getByText('151')).toBeInTheDocument()
  })
})
