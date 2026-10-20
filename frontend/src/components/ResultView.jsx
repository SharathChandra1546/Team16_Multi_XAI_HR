import { useState } from "react";
import { api, userId } from "../api.js";
import EvidenceChart from "./EvidenceChart.jsx";
import FollowupChat from "./FollowupChat.jsx";
import RiskSummary from "./RiskSummary.jsx";
import { BandBadge, ErrorText, Notice, pct } from "./ui.jsx";

const LEVEL_NAMES = { novice: "Novice", intermediate: "Intermediate", expert: "Expert" };

export function Verification({ r }) {
  if (!r.verification) return null;
  const v = r.verification;
  if (r.status === "fallback")
    return <Notice role="status">The language model's drafts did not pass the checks after {r.attempts} attempts, so a verified template is shown. All {v.n_claims} claims match the evidence.</Notice>;
  return (
    <p className="small" role="status">
      <span className="badge" style={{ marginRight: 8 }}>Verified</span>
      {v.n_claims} of {v.n_claims} claims match the evidence. Wording checks passed. Attempt {r.attempts} of 3.
    </p>
  );
}

export function Claims({ r }) {
  if (!r.explanation || !r.explanation.claims.length) return null;
  const gl = Object.fromEntries((r.evidence?.glossary || []).map((g) => [g.id, g]));
  return (
    <details>
      <summary>Claims and the evidence they cite</summary>
      <ol className="small" style={{ paddingLeft: 20 }}>
        {r.explanation.claims.map((c, i) => (
          <li key={i} style={{ marginBottom: 6 }}>
            {c.text} <span className="muted">[{c.evidence_ids.join(", ")}]</span>
          </li>
        ))}
      </ol>
      {Object.keys(gl).length > 0 && (
        <>
          <p className="small"><strong>Glossary</strong></p>
          <dl className="small">
            {Object.values(gl).map((g) => (
              <div key={g.id} style={{ marginBottom: 6 }}>
                <dt><strong>{g.term}</strong> <span className="muted">({g.id})</span></dt>
                <dd style={{ margin: 0 }}>{g.definition}{Object.keys(g.values || {}).length ? ` Values: ${Object.values(g.values).join(", ")}.` : ""}</dd>
              </div>
            ))}
          </dl>
        </>
      )}
    </details>
  );
}

export function Recourse({ r }) {
  const opts = r.evidence?.recourse || [];
  if (r.intent !== "recourse") return null;
  if (!opts.length) return <p className="muted">{r.evidence?.recourse_note}</p>;
  return (
    <div>
      <h3>Changes the model evaluated</h3>
      <p className="small muted">Each option was applied to this profile and re-run through the model. They are possibilities to discuss, not recommendations.</p>
      <div className="grid-3">
        {opts.map((o) => (
          <div className="card" key={o.id}>
            <p style={{ fontWeight: 700, marginBottom: 6 }}>{o.id.replace("R", "Option ")}</p>
            <p className="num" style={{ marginBottom: 6 }}>{pct(o.probability_before)} to {pct(o.probability_after)} <BandBadge band={o.band_after} /></p>
            <ul className="small" style={{ paddingLeft: 18, margin: 0 }}>
              {o.changes.map((c) => <li key={c.feature}>{c.display_name}: {c.from_label} to {c.to_label}{c.consequence ? " (follows from the change)" : ""}</li>)}
            </ul>
            {o.assumptions.map((a) => <p key={a} className="small muted" style={{ marginTop: 6 }}>Assumption: {a}</p>)}
          </div>
        ))}
      </div>
    </div>
  );
}

export function Trace({ r }) {
  if (!r.trace?.length) return null;
  return (
    <details>
      <summary>How this answer was produced</summary>
      <div className="table-wrap">
        <table className="small">
          <thead><tr><th>Step</th><th className="r">Time</th><th>Details</th></tr></thead>
          <tbody>
            {r.trace.map((t, i) => (
              <tr key={i}>
                <td>{t.node}</td>
                <td className="r num">{t.ms} ms</td>
                <td>{[t.intent && `intent ${t.intent}`, t.attempt && `attempt ${t.attempt}`, t.passed !== undefined && (t.passed ? "passed" : "failed"),
                  t.approved !== undefined && (t.approved ? "approved" : "rejected"), t.codes?.length ? t.codes.join(", ") : null].filter(Boolean).join("; ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">Model {r.prediction?.model_version}. Language model: {r.llm}. Prompt version: {r.prompt_version}. Trace ID {r.trace_id}.</p>
    </details>
  );
}

export function Feedback({ traceId }) {
  const [rating, setRating] = useState("");
  const [understood, setUnderstood] = useState("");
  const [trust, setTrust] = useState("");
  const [comment, setComment] = useState("");
  const [website, setWebsite] = useState("");
  const [sent, setSent] = useState(false);
  const [error, setError] = useState(null);
  const [formError, setFormError] = useState("");
  if (sent) return <Notice kind="ok" role="status">Thank you. Your rating was saved.</Notice>;
  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    if (!rating) return setFormError("Choose a rating from 1 to 5.");
    if (comment.length > 2000) return setFormError("Keep the comment under 2,000 characters.");
    setFormError("");
    try {
      await api("/feedback", { method: "POST", body: { trace_id: traceId, user_id: userId(), rating: Number(rating),
        understood: understood === "" ? null : understood === "yes", trust: trust ? Number(trust) : null, comment, website } });
      setSent(true);
    } catch (err) {
      setError(err);
    }
  };
  const scale = (name, value, set, label) => (
    <fieldset>
      <legend>{label}</legend>
      <div className="radio-row">
        {[1, 2, 3, 4, 5].map((n) => (
          <label key={n}><input type="radio" name={name} value={n} checked={value === String(n)} onChange={(e) => set(e.target.value)} /><span>{n}</span></label>
        ))}
      </div>
    </fieldset>
  );
  return (
    <form onSubmit={submit} noValidate aria-labelledby="fb-title">
      <h3 id="fb-title">Rate this explanation</h3>
      {scale("rating", rating, setRating, "How useful was it? (1 = not useful, 5 = very useful)")}
      <fieldset>
        <legend>Did you understand it?</legend>
        <div className="radio-row">
          {["yes", "no"].map((v) => (
            <label key={v}><input type="radio" name="understood" value={v} checked={understood === v} onChange={(e) => setUnderstood(e.target.value)} /><span>{v === "yes" ? "Yes" : "No"}</span></label>
          ))}
        </div>
      </fieldset>
      {scale("trust", trust, setTrust, "How much do you trust it? (optional)")}
      <div className="field">
        <label htmlFor="fb-comment">Comment (optional)</label>
        <textarea id="fb-comment" maxLength={2000} value={comment} onChange={(e) => setComment(e.target.value)} />
        <span className="hint">Do not include names or other personal details.</span>
      </div>
      <div className="hp" aria-hidden="true">
        <label htmlFor="fb-website">Leave this field empty</label>
        <input id="fb-website" tabIndex={-1} autoComplete="off" value={website} onChange={(e) => setWebsite(e.target.value)} />
      </div>
      {formError && <p className="form-error" role="alert">{formError}</p>}
      <ErrorText error={error} />
      <p style={{ marginTop: 12 }}><button className="btn secondary" type="submit">Send rating</button></p>
    </form>
  );
}

export default function ResultView({ r, onRefine, busy, consentCloud }) {
  if (!r) return null;
  if (r.status === "refused") return <Notice kind="danger" role="alert"><strong>Request refused.</strong> {r.message}</Notice>;
  if (r.status === "clarify" || r.status === "error") return <Notice kind="danger" role="alert">{r.message}</Notice>;
  if (r.status === "concept")
    return (
      <div className="card" aria-live="polite">
        <h2>Definition</h2>
        <p>{r.explanation.explanation}</p>
        <p className="small muted">Source: dataset data dictionary. Glossary ID {r.explanation.claims[0]?.evidence_ids[0]}.</p>
      </div>
    );
  return (
    <div className="stack" aria-live="polite">
      {r.prediction && <div className="card"><RiskSummary prediction={r.prediction} /></div>}
      {r.status === "escalated" && <Notice role="alert"><strong>Routed to human review.</strong> {r.message.replace(/^Routed to human review\.\s*/, "")}</Notice>}
      {r.status === "withheld" && <Notice kind="danger" role="alert">{r.message}</Notice>}
      {r.notices.filter((n) => !n.startsWith("Routed to human review")).map((n) => <Notice key={n} role="note">{n}</Notice>)}
      {r.explanation && (
        <div className="card">
          <h2 style={{ display: "flex", gap: 10, alignItems: "baseline", flexWrap: "wrap" }}>
            Explanation <span className="badge">{LEVEL_NAMES[r.level]}</span>
          </h2>
          <p>{r.explanation.explanation}</p>
          {r.disclosure && <p className="disclosure small">{r.disclosure}</p>}
          <Verification r={r} />
          {onRefine && (
            <div className="chips">
              <button className="btn small secondary" type="button" disabled={busy || r.level === "novice"} onClick={() => onRefine("simplify")}>Simpler</button>
              <button className="btn small secondary" type="button" disabled={busy || r.level === "expert"} onClick={() => onRefine("expand")}>More detail</button>
            </div>
          )}
        </div>
      )}
      {r.evidence && r.status !== "escalated" && r.status !== "withheld" && (
        <div className="card"><h2>Factors</h2><EvidenceChart items={r.evidence.items} max={r.level === "novice" ? 5 : 8} /></div>
      )}
      <Recourse r={r} />
      {r.explanation && <div className="card stack"><Claims r={r} /><Trace r={r} /></div>}
      {!r.explanation && <Trace r={r} />}
      {(r.status === "verified" || r.status === "fallback") && (
        <div className="card"><FollowupChat traceId={r.trace_id} consentCloud={consentCloud} /></div>
      )}
      {(r.status === "verified" || r.status === "fallback") && <div className="card"><Feedback traceId={r.trace_id} /></div>}
    </div>
  );
}
