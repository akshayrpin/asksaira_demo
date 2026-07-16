import { useRef, useState } from 'react'

// The fixed-five widget payloads emitted by the instant-permit apply agent
// (backend: apply_agent._widget_for). Rendered inline under the assistant's message.
export interface ApplyWidgetData {
  type: 'chips' | 'address_autocomplete' | 'form' | 'review' | 'result'
  field?: string
  options?: string[]
  fields?: { name: string; label: string; inputType?: string }[]
  confirm?: boolean
  data?: {
    applicant?: { name?: string; email?: string; phone?: string }
    property?: string
    job?: string
    work?: string
    valuation?: number
    fee?: number
    feeDetails?: { feeDescription?: string; feeAmount?: number }[]
  }
  permitNumber?: string
}

interface Props {
  widget: ApplyWidgetData
  onSubmit: (text: string) => void   // posts the tap/selection back as a normal user message
  disabled?: boolean
}

const chip: React.CSSProperties = {
  padding: '8px 14px', borderRadius: 20, border: '1px solid #0f6cbd',
  background: '#fff', color: '#0f6cbd', cursor: 'pointer', fontSize: 14, fontWeight: 600
}
const card: React.CSSProperties = {
  marginTop: 8, padding: 14, border: '1px solid #e1e1e1', borderRadius: 10, background: '#fff', maxWidth: 420
}
const input: React.CSSProperties = {
  width: '100%', padding: '10px 12px', border: '1px solid #c8c8c8', borderRadius: 8, fontSize: 14, boxSizing: 'border-box'
}
const primaryBtn: React.CSSProperties = {
  marginTop: 12, padding: '10px 16px', borderRadius: 8, border: 'none',
  background: '#0f6cbd', color: '#fff', cursor: 'pointer', fontSize: 14, fontWeight: 600
}

const Row = ({ label, value, bold }: { label: string; value?: string | number; bold?: boolean }) =>
  value === undefined || value === null || value === '' ? null : (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', fontWeight: bold ? 700 : 400 }}>
      <span style={{ color: '#666' }}>{label}</span>
      <span style={{ textAlign: 'right' }}>{value}</span>
    </div>
  )

function AddressAutocomplete({ onSubmit, disabled }: { onSubmit: (t: string) => void; disabled?: boolean }) {
  const [q, setQ] = useState('')
  const [opts, setOpts] = useState<{ address: string; lsoId: string | number }[]>([])
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Debounced: one request ~250ms after the user pauses, min 2 chars. Never per-keystroke.
  const onChange = (v: string) => {
    setQ(v)
    if (timer.current) clearTimeout(timer.current)
    if (v.trim().length < 2) { setOpts([]); return }
    timer.current = setTimeout(async () => {
      try {
        const r = await fetch(`/apply/address-search?q=${encodeURIComponent(v)}`)
        setOpts(r.ok ? await r.json() : [])
      } catch { setOpts([]) }
    }, 250)
  }

  return (
    <div style={{ marginTop: 8, maxWidth: 420, position: 'relative' }}>
      <input style={input} value={q} disabled={disabled} placeholder="Start typing the address…"
        onChange={e => onChange(e.target.value)} />
      {opts.length > 0 && (
        <div style={{ border: '1px solid #e1e1e1', borderRadius: 8, marginTop: 4, background: '#fff', overflow: 'hidden' }}>
          {opts.map(o => (
            <div key={String(o.lsoId)} style={{ padding: '8px 12px', cursor: 'pointer', fontSize: 14 }}
              onMouseDown={() => { setOpts([]); setQ(o.address); onSubmit(o.address) }}>
              {o.address}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function FormWidget({ fields, onSubmit, disabled }: {
  fields: { name: string; label: string; inputType?: string }[]; onSubmit: (t: string) => void; disabled?: boolean
}) {
  const [vals, setVals] = useState<Record<string, string>>({})
  const ready = fields.every(f => (vals[f.name] || '').trim())
  // Post the collected fields as one message; the agent extracts each value (name/email/phone/valuation).
  const submit = () => onSubmit(fields.map(f => `${f.name}: ${vals[f.name] || ''}`).join(', '))
  return (
    <div style={card}>
      {fields.map(f => (
        <div key={f.name} style={{ marginBottom: 8 }}>
          <label style={{ display: 'block', fontSize: 13, color: '#666', marginBottom: 2 }}>{f.label}</label>
          <input style={input} type={f.inputType || 'text'} value={vals[f.name] || ''} disabled={disabled}
            onChange={e => setVals(p => ({ ...p, [f.name]: e.target.value }))} />
        </div>
      ))}
      <button style={{ ...primaryBtn, opacity: ready ? 1 : 0.5 }} disabled={disabled || !ready} onClick={submit}>Continue</button>
    </div>
  )
}

export const ApplyWidget = ({ widget, onSubmit, disabled }: Props) => {
  if (!widget) return null

  if (widget.type === 'form') {
    return <FormWidget fields={widget.fields || []} onSubmit={onSubmit} disabled={disabled} />
  }

  if (widget.type === 'chips') {
    return (
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
        {(widget.options || []).map(o => (
          <button key={o} style={chip} disabled={disabled} onClick={() => onSubmit(o)}>{o}</button>
        ))}
      </div>
    )
  }

  if (widget.type === 'address_autocomplete') {
    return <AddressAutocomplete onSubmit={onSubmit} disabled={disabled} />
  }

  if (widget.type === 'review') {
    const d = widget.data || {}
    return (
      <div style={card}>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>Review your application</div>
        <Row label="Job" value={d.job} />
        <Row label="Address" value={d.property} />
        <Row label="Work" value={d.work} />
        <Row label="Valuation" value={d.valuation != null ? `$${d.valuation}` : undefined} />
        <Row label="Fee" value={d.fee != null ? `$${d.fee}` : undefined} bold />
        <button style={primaryBtn} disabled={disabled} onClick={() => onSubmit('CONFIRM')}>Confirm & submit</button>
      </div>
    )
  }

  if (widget.type === 'result') {
    return (
      <div style={card}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>Permit filed ✅</div>
        <div>Permit number: <b>{widget.permitNumber}</b></div>
      </div>
    )
  }

  return null
}

export default ApplyWidget
